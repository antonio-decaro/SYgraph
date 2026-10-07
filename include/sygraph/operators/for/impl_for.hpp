/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
namespace operators {

namespace compute {

namespace detail {

template<frontier::frontier_view FW, graph::detail::GraphConcept GraphT, typename InFrontierT>
inline sygraph::detail::kernel::LaunchConfig buildLaunchConfig(const GraphT& graph, const InFrontierT& in, int expected_size, sycl::queue& q) {
  sygraph::detail::kernel::LaunchConfig config{};
  auto in_dev_frontier = in.getDeviceFrontier();
  uint32_t active_size = 0;
  size_t requested_global = 0;

  if constexpr (FW != sygraph::frontier::frontier_view::vertex) { throw std::runtime_error("Invalid frontier view for compute operation."); }
  if (expected_size != frontier::size::fetch_from_memory) {
    throw std::runtime_error("Invalid expected_size value. Only fetch_from_memory is supported for compute operation.");
  }

  config.dependency = in.computeActiveFrontier();
  config.dependency.wait_and_throw();
  auto copy_e = q.copy(in_dev_frontier.getOffsetsSize(), &active_size, 1);
  copy_e.wait();
#ifdef ENABLE_PROFILING
  sygraph::Profiler::addEvent(copy_e, "frontier_size_fetch");
#endif
  const size_t bitmap_range = in.getBitmapRange();
  requested_global = static_cast<size_t>(active_size) * bitmap_range;

  config.local = {bitmap_range};
  config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};

  return config;
}

template<frontier::frontier_view FW, graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::Event launchBitmapKernel(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, frontier::frontier_type::mlb>& frontier,
                                  LambdaT&& functor,
                                  int expected_size) {
  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  auto config = buildLaunchConfig<FW>(graph, frontier, expected_size, q);

  size_t num_nodes = graph.getVertexCount();
  size_t bitmap_range = frontier.getBitmapRange();

  return q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>{config.global, config.local}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id)) { functor(actual_id); }
    });
  });
}

template<frontier::frontier_view FW, typename ReductionOperator, graph::detail::GraphConcept GraphT, typename T, typename R, typename LambdaT>
sygraph::Event launchBitmapReduce(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, frontier::frontier_type::mlb>& frontier,
                                  R& accumulator,
                                  LambdaT&& functor,
                                  int expected_size) {
  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  auto config = buildLaunchConfig<FW>(graph, frontier, expected_size, q);

  size_t num_nodes = graph.getVertexCount();
  size_t bitmap_range = frontier.getBitmapRange();

  sycl::buffer<R, 1> accumulator_buf(&accumulator, sycl::range<1>(1));
  accumulator_buf.set_final_data(&accumulator);
  accumulator_buf.set_write_back(true);

  return q.submit([&](sycl::handler& cgh) {
    auto reduction = sycl::reduction<R>(accumulator_buf, cgh, ReductionOperator{});
    cgh.parallel_for(sycl::nd_range<1>{config.global, config.local}, reduction, [=](sycl::nd_item<1> item, auto& acc) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id)) { functor(actual_id, acc); }
    });
  });
}

} // namespace detail
} // namespace compute
} // namespace operators
} // namespace sygraph
