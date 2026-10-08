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

namespace filter {
namespace detail {

template<graph::detail::GraphConcept GraphT, typename InFrontierT>
sygraph::detail::kernel::LaunchConfig buildLaunchConfig(const GraphT& graph, const InFrontierT& in, int expected_size, sycl::queue& q) {
  sygraph::detail::kernel::LaunchConfig config{};
  auto in_dev_frontier = in.getDeviceFrontier();
  uint32_t active_size = 0;
  size_t requested_global = 0;

  config.local = {in.getBitmapRange()};

  if (expected_size != frontier::size::fetch_from_memory) {
    throw std::runtime_error("Invalid expected_size value. Only fetch_from_memory is supported for filter operation.");
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
  config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  return config;
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event
launchBitmapKernelExternal(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& in, sygraph::frontier::Frontier<T, FT>& out, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::mlb) {
    throw std::runtime_error("Invalid frontier type");
  }

  auto q = graph.getQueue();

  size_t num_nodes = graph.getVertexCount();

  auto config = buildLaunchConfig(graph, in, frontier::size::fetch_from_memory, q);

  out.clear();

  using type_t = T;

  auto out_dev = out.getDeviceFrontier();
  auto in_dev = in.getDeviceFrontier();
  uint8_t bitmap_range = in_dev.getBitmapRange();

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>{config.global, config.local}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = in_dev.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && in_dev.check(actual_id) && functor(actual_id)) { out_dev.insert(actual_id); }
    });
  });

  return e;
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FT, typename LambdaT>
sygraph::Event launchBitmapKernelInplace(GraphT& graph, const sygraph::frontier::Frontier<T, FT>& frontier, LambdaT&& functor) {
  if constexpr (FT != sygraph::frontier::frontier_type::bitmap && FT != sygraph::frontier::frontier_type::mlb) {
    throw std::runtime_error("Invalid frontier type");
  }

  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  size_t num_nodes = graph.getVertexCount();

  auto config = buildLaunchConfig(graph, frontier, frontier::size::fetch_from_memory, q);

  size_t bitmap_range = frontier.getBitmapRange();

  using type_t = T;

  sygraph::Event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(sycl::nd_range<1>{config.global, config.local}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id) && functor(actual_id)) { dev_frontier.remove(actual_id); }
    });
  });

  return e;
}

} // namespace detail
} // namespace filter
} // namespace operators
} // namespace sygraph