/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <type_traits>

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/device.hpp>
#include <sygraph/utils/kernel.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace operators {
namespace advance {
namespace detail {

struct AdvanceContextState {
  size_t group_offset;
  uint16_t coarsening_factor;
  uint32_t offsets_size;
  sycl::nd_item<1> item;
};

template<sygraph::frontier::frontier_view IFW,
         sygraph::frontier::frontier_view OFW,
         sygraph::operators::direction Direction,
         typename InFrontierDevT,
         typename OutFrontierDevT>
struct AdvanceContextBase {
  size_t limit;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;

  AdvanceContextBase(size_t limit, InFrontierDevT in_dev_frontier, OutFrontierDevT out_dev_frontier)
      : limit(limit), in_dev_frontier(in_dev_frontier), out_dev_frontier(out_dev_frontier) {}

  SYCL_EXTERNAL inline bool needToProcess(const AdvanceContextState& state) const {
    return (state.group_offset * state.coarsening_factor < state.offsets_size);
  }

  SYCL_EXTERNAL inline void completeIteration(AdvanceContextState& state) const { state.group_offset += state.item.get_group_range(0); }

  SYCL_EXTERNAL inline bool check(const AdvanceContextState&, const uint32_t& vertex) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return vertex < limit && ((Direction == sygraph::operators::direction::push) == in_dev_frontier.check(vertex));
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return vertex < limit;
    } else {
      return false;
    }
  }

  SYCL_EXTERNAL inline bool isValidNeighbor(const AdvanceContextState&, const uint32_t& neighbor) const {
    if constexpr (sygraph::operators::is_pull<Direction>()) {
      return in_dev_frontier.check(neighbor);
    } else {
      return true;
    }
  }

  SYCL_EXTERNAL inline void insert(const AdvanceContextState&, const uint32_t& vertex, const uint32_t& neighbor) const {
    if constexpr (OFW == sygraph::frontier::frontier_view::vertex) {
      if constexpr (!sygraph::operators::is_pull<Direction>()) {
        out_dev_frontier.insert(neighbor);
      } else {
        out_dev_frontier.insert(vertex);
      }
    }
  }
};

template<typename GraphDevT, typename InFrontierDevT, typename OutFrontierDevT>
struct AdvanceLaunchSetup {
  sycl::queue& q;
  size_t num_nodes;
  InFrontierDevT in_dev_frontier;
  OutFrontierDevT out_dev_frontier;
  GraphDevT graph_dev;
  size_t coarsening_factor;
  sygraph::detail::kernel::LaunchConfig launch_config;
};

template<sygraph::frontier::frontier_view InFW, typename GraphT, typename InFrontierT>
inline sygraph::detail::kernel::LaunchConfig
buildAdvanceLaunchConfig(GraphT& graph, const InFrontierT& in, bool pull_advance, int expected_size, size_t coarsening_factor, sycl::queue& q) {
  sygraph::detail::kernel::LaunchConfig config{};
  auto in_dev_frontier = in.getDeviceFrontier();
  if constexpr (InFW == sygraph::frontier::frontier_view::vertex) {
    const size_t bitmap_range = in.getBitmapRange();
    config.local = {bitmap_range * coarsening_factor};
    uint32_t active_size = 0;
    if constexpr (requires { in.computeActiveFrontier(pull_advance); }) {
      config.dependency = in.computeActiveFrontier(pull_advance);
    } else {
      active_size = static_cast<uint32_t>(in.computeActiveFrontier());
    }

    size_t requested_global = 0;
    if (expected_size > 0) {
      requested_global = static_cast<size_t>(expected_size);
    } else if (expected_size == frontier::size::infer_from_device) {
      // `coarsening_factor` is already encoded in the work-group width:
      // each work-group covers `coarsening_factor` bitmap-offset integers.
      requested_global = config.local[0] * sygraph::detail::device::getNumComputeUnits(q);
    } else if (expected_size == frontier::size::fetch_from_memory) {
      if constexpr (requires { in.computeActiveFrontier(pull_advance); }) {
        config.dependency.wait_and_throw();
        auto copy_e = q.copy(in_dev_frontier.getOffsetsSize(), &active_size, 1);
        copy_e.wait();
#ifdef ENABLE_PROFILING
        sygraph::Profiler::addEvent(copy_e, "frontier_size_fetch");
#endif
      }
      requested_global = static_cast<size_t>(active_size) * bitmap_range;
    } else {
      throw std::runtime_error("Invalid expected_size value");
    }

    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else if constexpr (InFW == sygraph::frontier::frontier_view::graph) {
    config.local = {types::detail::COMPUTE_UNIT_SIZE};
    const size_t requested_global = graph.getVertexCount();
    config.global = {sygraph::detail::kernel::ensureLocalMultiple(requested_global, config.local[0])};
  } else {
    throw std::runtime_error("Invalid frontier view for compute operation.");
  }

  return config;
}

template<sygraph::frontier::frontier_view InFW,
         sygraph::operators::direction Direction,
         graph::detail::GraphConcept GraphT,
         typename InFrontierT,
         typename OutFrontierT>
inline auto prepareAdvanceLaunch(GraphT& graph, const InFrontierT& in, const OutFrontierT& out, int expected_size) {
  // Pull visits the vertices outside the frontier, which needs the inverted active frontier only MLB provides.
  static_assert(
      !sygraph::operators::is_pull<Direction>() || InFW != sygraph::frontier::frontier_view::vertex || requires { in.computeActiveFrontier(true); },
      "Pull advance requires an MLB frontier; the bitmap frontier does not support it.");
  sycl::queue& q = graph.getQueue();
  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();
  if constexpr (sygraph::operators::is_pull<Direction>()) { graph_dev = graph.getInverseDeviceGraph(); }

  const size_t coarsening_factor = types::detail::COMPUTE_UNIT_SIZE / sygraph::detail::device::getSubgroupSize(q);
  const bool pull_advance = sygraph::operators::is_pull<Direction>();
  auto launch_config = buildAdvanceLaunchConfig<InFW>(graph, in, pull_advance, expected_size, coarsening_factor, q);

  return AdvanceLaunchSetup<decltype(graph_dev), decltype(in_dev_frontier), decltype(out_dev_frontier)>{
      q, graph.getVertexCount(), in_dev_frontier, out_dev_frontier, graph_dev, coarsening_factor, launch_config};
}

template<sygraph::frontier::frontier_view InFW, typename GraphT>
using advance_element_t = std::conditional_t<InFW == sygraph::frontier::frontier_view::vertex, typename GraphT::vertex_t, typename GraphT::edge_t>;

} // namespace detail
} // namespace advance
} // namespace operators
} // namespace sygraph
