/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/operators/advance/common.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/device.hpp>

namespace sygraph {
namespace operators {

namespace advance {

namespace detail {

template<sygraph::frontier::frontier_view IFW,
         sygraph::frontier::frontier_view OFW,
         sygraph::operators::direction Direction,
         typename InFrontierDevT,
         typename OutFrontierDevT>
struct BucketingContext : AdvanceContextBase<IFW, OFW, Direction, InFrontierDevT, OutFrontierDevT> {
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
      return state.item.get_global_linear_id();
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
struct BitmapKernel {
  // Entry point invoked by the SYCL runtime for each work-item.
  void operator()(sycl::nd_item<1> item) const {
    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const size_t wgroup_size = wgroup.get_local_range(0);
    const auto sgroup = item.get_sub_group();
    const auto sgroup_id = sgroup.get_group_id();
    const size_t sgroup_size = sgroup.get_local_range()[0];
    const size_t llid = sgroup.get_local_linear_id();

    auto state = context.init(item);

    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{subgroup_reduce_tail[sgroup_id]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{workgroup_reduce_tail[0]};

    while (context.needToProcess(state)) {
      const auto assigned_vertex = context.getAssignedElement(state);

      // Reset local reductions so the group can classify work for this iteration.
      if (sgroup.leader()) { subgroup_reduce_tail[sgroup_id] = 0; }
      if (wgroup.leader()) { workgroup_reduce_tail[0] = 0; }
      workgroup_claimed[lid] = 0;
      subgroup_claimed[lid] = 0;


      const uint32_t offset = sgroup_id * sgroup_size;
      if (context.check(state, assigned_vertex)) {
        const uint32_t n_edges = graph_dev.getDegree(assigned_vertex);
        if (n_edges >= wgroup_size * wgroup_size) { // dispatch to the whole workgroup
          const uint32_t loc = wg_tail.fetch_add(1U);
          n_edges_wg[loc] = n_edges;
          workgroup_reduce[loc] = assigned_vertex;
          workgroup_ids[loc] = lid;
        } else if (n_edges >= sgroup_size) { // dispatch to a subgroup
          const uint32_t loc = sg_tail.fetch_add(1U);
          n_edges_sg[offset + loc] = n_edges;
          subgroup_reduce[offset + loc] = assigned_vertex;
          subgroup_ids[offset + loc] = lid;
        }
        visited[lid] = false;
      } else {
        visited[lid] = true;
      }

      // Process vertices assigned to the workgroup.
      sycl::group_barrier(wgroup);
      const auto wg_assigned = wg_tail.load();
      for (size_t i = 0; i < wg_assigned; i++) {
        const auto vertex = workgroup_reduce[i];
        const size_t n_edges = n_edges_wg[i];
        auto start = graph_dev.begin(vertex);

        for (auto j = lid; j < n_edges; j += wgroup_size) {
          if (shouldShortCircuitWorkgroup(i)) { break; }
          auto n = start + j;
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          processEdge(state, vertex, neighbor, edge, weight, workgroup_claimed, i);
        }

        if (wgroup.leader()) { visited[workgroup_ids[i]] = true; }
      }

      // Process vertices assigned to subgroups.
      sycl::group_barrier(sgroup);
      const auto sg_assigned = sg_tail.load();
      for (size_t i = 0; i < sg_assigned; i++) {
        const size_t vertex_id = offset + i;
        const auto vertex = subgroup_reduce[vertex_id];
        const size_t n_edges = n_edges_sg[vertex_id];
        auto start = graph_dev.begin(vertex);

        for (auto j = llid; j < n_edges; j += sgroup_size) {
          if (shouldShortCircuitSubgroup(vertex_id)) { break; }
          auto n = start + j;
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          processEdge(state, vertex, neighbor, edge, weight, subgroup_claimed, vertex_id);
        }

        if (sgroup.leader()) { visited[subgroup_ids[vertex_id]] = true; }
      }
      sycl::group_barrier(sgroup);

      // Fallback: each lane processes its own very small vertex.
      if (!visited[lid]) {
        const auto vertex = assigned_vertex;
        auto start = graph_dev.begin(vertex);
        const auto end = graph_dev.end(vertex);

        for (auto n = start; n != end; ++n) {
          const auto edge = n.getIndex();
          const auto weight = graph_dev.getEdgeWeight(edge);
          const auto neighbor = *n;
          if (!context.isValidNeighbor(state, neighbor)) { continue; }
          if (!functor(vertex, neighbor, edge, weight)) { continue; }
          context.insert(state, vertex, neighbor);
          if (shouldShortCircuitLane()) { break; }
        }
      }
      context.completeIteration(state);
    }
  }

  const ContextT context;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> n_edges_wg;
  const sycl::local_accessor<uint32_t, 1> n_edges_sg;
  const sycl::local_accessor<bool, 1> visited;
  const sycl::local_accessor<T, 1> subgroup_reduce;
  const sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail;
  const sycl::local_accessor<uint32_t, 1> subgroup_ids;
  const sycl::local_accessor<uint32_t, 1> subgroup_claimed;
  const sycl::local_accessor<T, 1> workgroup_reduce;
  const sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail;
  const sycl::local_accessor<uint32_t, 1> workgroup_ids;
  const sycl::local_accessor<uint32_t, 1> workgroup_claimed;
  const LambdaT functor;

  template<typename WeightT, typename FlagAccessorT>
  SYCL_EXTERNAL inline bool processEdge(const AdvanceContextState& state,
                                        const uint32_t vertex,
                                        const uint32_t neighbor,
                                        const uint32_t edge,
                                        const WeightT& weight,
                                        const FlagAccessorT& flags,
                                        const uint32_t flag_index) const {
    if (!context.isValidNeighbor(state, neighbor)) { return false; }
    if (!functor(vertex, neighbor, edge, weight)) { return false; }
    context.insert(state, vertex, neighbor);
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> claimed(flags[flag_index]);
      claimed.store(1);
    }
    return true;
  }

  SYCL_EXTERNAL inline bool shouldShortCircuitWorkgroup(const uint32_t slot) const {
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> claimed(workgroup_claimed[slot]);
      return claimed.load() != 0;
    }
    return false;
  }

  SYCL_EXTERNAL inline bool shouldShortCircuitSubgroup(const uint32_t slot) const {
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) {
      sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> claimed(subgroup_claimed[slot]);
      return claimed.load() != 0;
    }
    return false;
  }

  SYCL_EXTERNAL inline bool shouldShortCircuitLane() const {
    if constexpr (sygraph::operators::is_short_circuit<Direction>()) { return true; }
    return false;
  }
};

namespace bucketing {

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
  BucketingContext<InFW, OutFW, Direction, decltype(launch.in_dev_frontier), decltype(launch.out_dev_frontier)> context{
      launch.num_nodes, launch.in_dev_frontier, launch.out_dev_frontier};
  const uint32_t max_num_subgroups = sygraph::detail::device::getMaxNumSubgroups(launch.q);
  using bitmap_kernel_t = BitmapKernel<InFW, OutFW, Direction, element_t, decltype(context), decltype(launch.graph_dev), LambdaT>;

  auto e = launch.q.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependency);
    // Local storage used to distribute vertices across workgroups/subgroups.
    sycl::local_accessor<uint32_t, 1> n_edges_wg{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> n_edges_sg{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<element_t, 1> subgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_reduce_tail{max_num_subgroups, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_ids{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> subgroup_claimed{local_range, cgh};
    sycl::local_accessor<element_t, 1> workgroup_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_reduce_tail{1, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_ids{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> workgroup_claimed{local_range, cgh};

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                     bitmap_kernel_t{context,
                                     launch.graph_dev,
                                     n_edges_wg,
                                     n_edges_sg,
                                     visited,
                                     subgroup_reduce,
                                     subgroup_reduce_tail,
                                     subgroup_ids,
                                     subgroup_claimed,
                                     workgroup_reduce,
                                     workgroup_reduce_tail,
                                     workgroup_ids,
                                     workgroup_claimed,
                                     std::forward<LambdaT>(functor)});
  });
  return {e};
}

} // namespace bucketing
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace sygraph
