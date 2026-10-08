/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/operators/advance/common.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>

namespace sygraph {
namespace operators {

namespace advance {

namespace detail {

template<typename T>
SYCL_EXTERNAL inline uint32_t workgroupMappedUpperBound(const T* values, uint32_t n, T value) {
  uint32_t left = 0;
  uint32_t right = n;
  while (left < right) {
    const uint32_t mid = left + ((right - left) >> 1);
    if (values[mid] <= value) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left;
}

template<sygraph::frontier::frontier_view IFW,
         sygraph::frontier::frontier_view OFW,
         sygraph::operators::direction Direction,
         typename InFrontierDevT,
         typename OutFrontierDevT>
struct WorkgroupMappedContext : AdvanceContextBase<IFW, OFW, Direction, InFrontierDevT, OutFrontierDevT> {
  using Base = AdvanceContextBase<IFW, OFW, Direction, InFrontierDevT, OutFrontierDevT>;
  using Base::Base;

  SYCL_EXTERNAL inline AdvanceContextState init(sycl::nd_item<1>& item) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      return {
          item.get_group_linear_id(),
          static_cast<uint16_t>(item.get_local_range(0) / this->in_dev_frontier.getBitmapRange()),
          this->in_dev_frontier.getOffsetsSize()[0],
          item,
      };
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return {
          item.get_group_linear_id(),
          static_cast<uint16_t>(item.get_local_range(0)),
          static_cast<uint32_t>(this->limit),
          item,
      };
    } else {
      return {0, 0, 0, item};
    }
  }

  SYCL_EXTERNAL inline size_t getAssignedElement(const AdvanceContextState& state) const {
    if constexpr (IFW == sygraph::frontier::frontier_view::vertex) {
      const uint16_t bitmap_range = this->in_dev_frontier.getBitmapRange();
      const uint32_t actual_id_offset = (state.group_offset * state.coarsening_factor) + (state.item.get_local_linear_id() / bitmap_range);
      if (actual_id_offset >= state.offsets_size) { return this->limit; }
      const int* bitmap_offsets = this->in_dev_frontier.getOffsets();
      const auto assigned_vertex = (bitmap_offsets[actual_id_offset] * bitmap_range) + (state.item.get_local_linear_id() % bitmap_range);
      return assigned_vertex;
    } else if constexpr (IFW == sygraph::frontier::frontier_view::graph) {
      return (state.group_offset * state.coarsening_factor) + state.item.get_local_linear_id();
    } else {
      return this->limit;
    }
  }
};

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         sygraph::operators::direction Direction,
         typename T,
         typename ContextT,
         graph::detail::DeviceGraphConcept GraphDevT,
         typename LambdaT>
// Work-distribution kernel that maps bitmap entries to the most suitable execution granularity.
struct WorkgroupMappedBitmapKernel {
  // Entry point invoked by the SYCL runtime for each work-item.
  void operator()(sycl::nd_item<1> item) const {
    static_assert(InFW != sygraph::frontier::frontier_view::none, "Workgroup-mapped advance requires an input frontier.");

    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const size_t wgroup_size = wgroup.get_local_range(0);

    auto state = context.init(item);

    while (context.needToProcess(state)) {
      const uint32_t assigned_vertex = static_cast<uint32_t>(context.getAssignedElement(state));
      const bool vertex_active = context.check(state, assigned_vertex);
      const uint32_t degree = vertex_active ? static_cast<uint32_t>(graph_dev.getDegree(assigned_vertex)) : 0U;

      vertices[lid] = vertex_active ? assigned_vertex : UINT32_MAX;
      start_edges[lid] = vertex_active ? static_cast<uint32_t>(graph_dev.getFirstNeighbor(assigned_vertex)) : 0U;

      const uint32_t inclusive_end = sycl::inclusive_scan_over_group(wgroup, degree, sycl::plus<uint32_t>());
      const uint32_t exclusive_begin = inclusive_end - degree;

      scan_begins[lid] = exclusive_begin;
      scan_ends[lid] = inclusive_end;
      source_done[lid] = 0;
      sycl::group_barrier(wgroup);
      // The last lane's inclusive prefix sum is the total number of edges assigned to the work-group.
      const uint32_t total_edges = scan_ends[wgroup_size - 1];

      for (uint32_t edge_rank = static_cast<uint32_t>(lid); edge_rank < total_edges; edge_rank += static_cast<uint32_t>(wgroup_size)) {
        const uint32_t slot = workgroupMappedUpperBound(&scan_ends[0], static_cast<uint32_t>(wgroup_size), edge_rank);
        if (slot >= wgroup_size) { continue; }
        if (shouldShortCircuitSlot(slot)) { continue; }

        const uint32_t source = vertices[slot];
        if (source == UINT32_MAX) { continue; }

        const uint32_t local_edge_offset = edge_rank - scan_begins[slot];
        const uint32_t edge = start_edges[slot] + local_edge_offset;
        if (edge >= graph_dev.getEdgeCount()) { continue; }

        const auto weight = graph_dev.getEdgeWeight(edge);
        const auto neighbor = graph_dev.getDestinationVertex(edge);
        processEdge(state, source, neighbor, edge, weight, slot);
      }

      sycl::group_barrier(wgroup);
      context.completeIteration(state);
    }
  }

  const ContextT context;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> vertices;
  const sycl::local_accessor<uint32_t, 1> start_edges;
  const sycl::local_accessor<uint32_t, 1> scan_begins;
  const sycl::local_accessor<uint32_t, 1> scan_ends;
  const sycl::local_accessor<uint32_t, 1> source_done;
  const LambdaT functor;

  SYCL_EXTERNAL inline bool shouldShortCircuitSlot(const uint32_t slot) const {
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> slot_done(source_done[slot]);
      return slot_done.load() != 0;
    } else {
      return false;
    }
  }

  template<typename WeightT>
  SYCL_EXTERNAL inline bool processEdge(const AdvanceContextState& state,
                                        const uint32_t source,
                                        const uint32_t neighbor,
                                        const uint32_t edge,
                                        const WeightT& weight,
                                        const uint32_t slot) const {
    if (!context.isValidNeighbor(state, neighbor)) { return false; }
    if (!functor(source, neighbor, edge, weight)) { return false; }
    context.insert(state, source, neighbor);
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> slot_done(source_done[slot]);
      slot_done.store(1);
    }
    return true;
  }
};

namespace workgroup_mapped {

template<sygraph::frontier::frontier_view InFW,
         sygraph::frontier::frontier_view OutFW,
         sygraph::operators::direction Direction,
         typename T,
         graph::detail::GraphConcept GraphT,
         typename InFrontierT,
         typename OutFrontierT,
         typename LambdaT>
// Launch the mapped advance kernel for the requested frontier/configuration.
sygraph::Event launchBitmapKernel(GraphT& graph, const InFrontierT& in, const OutFrontierT& out, LambdaT&& functor, int expected_size) {
  auto launch = prepareAdvanceLaunch<InFW, Direction>(graph, in, out, expected_size);
  const sycl::range<1>& local_range = launch.launch_config.local;
  const sycl::range<1>& global_range = launch.launch_config.global;
  const sycl::event& dependency = launch.launch_config.dependency;

  using element_t = advance_element_t<InFW, GraphT>;
  WorkgroupMappedContext<InFW, OutFW, Direction, decltype(launch.in_dev_frontier), decltype(launch.out_dev_frontier)> context{
      launch.num_nodes, launch.in_dev_frontier, launch.out_dev_frontier};
  using bitmap_kernel_t = WorkgroupMappedBitmapKernel<InFW, OutFW, Direction, element_t, decltype(context), decltype(launch.graph_dev), LambdaT>;

  auto e = launch.q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependency);
    sycl::local_accessor<uint32_t, 1> vertices{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> start_edges{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> scan_begins{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> scan_ends{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> source_done{local_range, cgh};

    // Unnamed kernel: its type includes the user functor, so every instantiation gets a distinct name.
    cgh.parallel_for(
        sycl::nd_range<1>{global_range, local_range},
        bitmap_kernel_t{context, launch.graph_dev, vertices, start_edges, scan_begins, scan_ends, source_done, std::forward<LambdaT>(functor)});
  });
  return {e};
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace sygraph
