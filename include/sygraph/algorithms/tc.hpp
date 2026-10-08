/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sygraph/operators/advance/advance.hpp"
#include "sygraph/operators/config.hpp"
#include "sygraph/utils/memory.hpp"
#include <sycl/sycl.hpp>

#include <memory>

#include <sygraph/graph/graph.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <sygraph/sync/atomics.hpp>


namespace sygraph {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct TCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;

  TCInstance(GraphType& G) : G(G) {
    sycl::queue& queue = G.getQueue();
    size_t num_nodes = G.getVertexCount();
    triangles = memory::detail::memoryAlloc<uint32_t, memory::space::device>(num_nodes, queue);
    queue.fill(triangles, static_cast<uint32_t>(0), num_nodes).wait();
  }

  ~TCInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(triangles, queue);
  }


  uint32_t* triangles;
};
} // namespace detail


template<typename GraphType>
class TC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  TC(GraphType& g) : _g(g) {};


  void init() { _instance = std::make_unique<detail::TCInstance<GraphType>>(_g); }


  void reset() { _instance.reset(); }


  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }

    auto& G = _instance->G;
    auto& triangles = _instance->triangles;

    sycl::queue& queue = G.getQueue();

    size_t num_nodes = G.getVertexCount();
    auto graph_dev = G.getDeviceGraph();

    constexpr auto lb = sygraph::operators::load_balancer::workgroup_mapped;
    auto e = sygraph::operators::advance::vertices<lb>(G, [=](auto u, auto v, auto e, auto w) {
      (void)e;
      (void)w;

      if (u >= v) { return false; }

      auto src_it = graph_dev.begin(u);
      auto src_end = graph_dev.end(u);
      auto dst_it = graph_dev.begin(v);
      auto dst_end = graph_dev.end(v);

      int local_triangles = 0;

      while (src_it != src_end && dst_it != dst_end) {
        if (*src_it == *dst_it) {
          ++local_triangles;
          ++src_it;
          ++dst_it;
        } else if (*src_it < *dst_it) {
          ++src_it;
        } else {
          ++dst_it;
        }
      }

      if (local_triangles > 0) { sygraph::sync::atomicFetchAdd<uint32_t>(triangles + u, local_triangles); }
      return false;
    });

    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "TC");
#endif
  }

  size_t getNumTriangles(vertex_t v) const {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }
    uint32_t count;
    _g.getQueue().copy(_instance->triangles + v, &count, 1).wait();
    return count;
  }

  size_t getNumTriangles() const {
    if (!_instance) { throw std::runtime_error("TC instance not initialized"); }

    sycl::queue& queue = _g.getQueue();
    size_t num_nodes = _g.getVertexCount();
    auto& triangles = _instance->triangles;

    sycl::buffer<uint32_t, 1> sum_buff(sycl::range<1>(1));

    queue
        .submit([&](sycl::handler& cgh) {
          auto red = sycl::reduction(sum_buff, cgh, sycl::plus<uint32_t>(), sycl::property::reduction::initialize_to_identity{});
          cgh.parallel_for(sycl::range{num_nodes}, red, [=](sycl::id<1> idx, auto& sum) { sum += triangles[idx]; });
        })
        .wait();

    sycl::host_accessor sum_acc(sum_buff, sycl::read_only);
    return sum_acc[0] / 3;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::TCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
