#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

template<typename T, typename FrontierDevT, graph::detail::DeviceGraphConcept GraphDevT, typename LambdaT>
struct vector_kernel {
  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    const size_t gid = item.get_global_linear_id();
    const size_t lid = item.get_local_linear_id();
    const size_t local_range = item.get_local_range(0);
    const auto group = item.get_group();
    const auto group_id = item.get_group_linear_id();
    const auto subgroup = item.get_sub_group();
    const auto subgroup_id = subgroup.get_group_id();
    const size_t subgroup_size = subgroup.get_local_range()[0];
    const size_t llid = subgroup.get_local_linear_id();

    uint32_t* global_tail = out_dev_frontier.getVectorSizePtr();

    // 1. load number of edges in local memory
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> pad_tail_ref{pad_tail[0]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> active_elems_tail_ref{
        active_elements_local_tail[subgroup_id]};

    if (gid == 0) { global_tail[0] = 0; }
    if (group.leader()) { pad_tail_ref.store(0); }
    if (subgroup.leader()) { active_elems_tail_ref.store(0); }

    const uint32_t offset = subgroup_id * subgroup_size;
    if (gid < active_elements_size) {
      T element = active_elements[gid];
      uint32_t n_edges = graph_dev.getDegree(element);
      if (n_edges >= subgroup_size) {
        uint32_t loc = active_elems_tail_ref.fetch_add(1);
        active_elements_local[offset + loc] = element;
        n_edges_local[offset + loc] = n_edges;
        ids[offset + loc] = lid;
      }
      // active_elements_local[lid] = element;
      visited[lid] = false;
    } else {
      n_edges_local[lid] = 0;
      visited[lid] = true;
    }
    sycl::group_barrier(subgroup);

    // 2. process elements with less than local_range edges but more than one subgroup size edges
    for (uint32_t i = 0; i < active_elems_tail_ref.load(); i++) {
      size_t vertex_id = offset + i;
      auto vertex = active_elements_local[vertex_id];
      size_t n_edges = n_edges_local[vertex_id];
      size_t private_slice = n_edges / subgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * llid);
      auto end = llid == subgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor, pad, pad_tail_ref); }
      }
      if (subgroup.leader()) { visited[ids[vertex_id]] = true; }
      // if (n_edges_local[vertex_id] >= subgroup_size * subgroup_size) {
      // }
    }
    sycl::group_barrier(group);

    // 3. process the rest
    if (!visited[lid]) {
      auto vertex = active_elements[gid];
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor, pad, pad_tail_ref); }
      }
    }

    sycl::group_barrier(group);
    out_dev_frontier.finalize(item, pad, pad_tail_ref);
  }

  const T* active_elements;
  const size_t active_elements_size;
  const FrontierDevT in_dev_frontier;
  const FrontierDevT out_dev_frontier;
  const GraphDevT graph_dev;
  const sycl::local_accessor<uint32_t, 1> n_edges_local;
  const sycl::local_accessor<bool, 1> visited;
  const sycl::local_accessor<T, 1> active_elements_local;
  const sycl::local_accessor<uint32_t, 1> active_elements_local_tail;
  const sycl::local_accessor<uint32_t, 1> ids;
  const sycl::local_accessor<T, 1> pad;
  const sycl::local_accessor<uint32_t, 1> pad_tail;
  const LambdaT functor;
};


template<typename T, typename FrontierDevT, graph::detail::DeviceGraphConcept GraphDevT, typename LambdaT>
struct bitmap_kernel {
  void operator()(sycl::nd_item<1> item) const {
    // 0. retrieve global and local ids
    const size_t gid = item.get_global_linear_id();
    const size_t lid = item.get_local_linear_id();
    const auto wgroup = item.get_group();
    const auto wgroup_id = item.get_group_linear_id();
    const size_t wgroup_size = wgroup.get_local_range(0);
    const auto sgroup = item.get_sub_group();
    const auto sgroup_id = sgroup.get_group_id();
    const size_t sgroup_size = sgroup.get_local_range()[0];
    const size_t llid = sgroup.get_local_linear_id();
    const int* bitmap_offsets = in_dev_frontier.getOffsets();
    const size_t bitmap_range = in_dev_frontier.getBitmapRange();

    const size_t coarsening_factor = wgroup_size / bitmap_range;
    const size_t acutal_id_offset = wgroup_id * coarsening_factor + lid / bitmap_range;
    const size_t assigned_vertex = bitmap_offsets[acutal_id_offset] * bitmap_range + lid % bitmap_range;

    // 1. load number of edges in local memory
    if (sgroup.leader()) { active_elements_tail[sgroup_id] = 0; }
    if (wgroup.leader()) { work_group_reduce_tail[0] = 0; }

    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::sub_group> sg_tail{active_elements_tail[sgroup_id]};
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> wg_tail{work_group_reduce_tail[0]};

    const size_t offset = sgroup_id * sgroup_size;
    if (assigned_vertex < num_nodes && in_dev_frontier.check(assigned_vertex)) {
      size_t n_edges = graph_dev.getDegree(assigned_vertex);
      if (n_edges >= wgroup_size * wgroup_size * wgroup_size) { // assign to the workgroup
        size_t loc = wg_tail.fetch_add(1);
        work_group_reduce[loc] = assigned_vertex;
      } else if (n_edges >= sgroup_size) { // assign to the subgroup
        size_t loc = sg_tail.fetch_add(1);
        n_edges_local[offset + loc] = n_edges;
        active_elements_local[offset + loc] = assigned_vertex;
      }
      // size_t loc = sg_tail.fetch_add(1);
      // n_edges_local[offset + loc] = n_edges;
      // active_elements_local[offset + loc] = assigned_vertex;
      visited[lid] = false;
    } else {
      visited[lid] = true;
    }

    sycl::group_barrier(wgroup);
    for (size_t i = 0; i < wg_tail.load(); i++) {
      auto vertex = work_group_reduce[i];
      size_t n_edges = graph_dev.getDegree(vertex);
      size_t private_slice = n_edges / wgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * lid);
      auto end = lid == wgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
      if (wgroup.leader()) { visited[vertex % wgroup_size] = true; }
    }

    sycl::group_barrier(sgroup);

    for (size_t i = 0; i < active_elements_tail[sgroup_id]; i++) { // active_elements_tail[subgroup_id] is always less or equal than subgroup_size
      size_t vertex_id = offset + i;
      auto vertex = active_elements_local[vertex_id];
      size_t n_edges = n_edges_local[vertex_id];
      // if (n_edges < sgroup_size) { continue; }
      size_t private_slice = n_edges / sgroup_size;
      auto start = graph_dev.begin(vertex) + (private_slice * llid);
      auto end = llid == sgroup_size - 1 ? graph_dev.end(vertex) : start + private_slice;

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
      if (sgroup.leader()) { visited[vertex % wgroup_size] = true; }
    }
    sycl::group_barrier(sgroup);

    if (!visited[lid]) {
      auto vertex = assigned_vertex;
      auto start = graph_dev.begin(vertex);
      auto end = graph_dev.end(vertex);

      for (auto n = start; n != end; ++n) {
        auto edge = n.get_index();
        auto weight = graph_dev.getEdgeWeight(edge);
        auto neighbor = *n;
        if (functor(vertex, neighbor, edge, weight)) { out_dev_frontier.insert(neighbor); }
      }
    }
  }

  size_t num_nodes;
  FrontierDevT in_dev_frontier;
  FrontierDevT out_dev_frontier;
  GraphDevT graph_dev;
  sycl::local_accessor<uint32_t, 1> n_edges_local;
  sycl::local_accessor<T, 1> active_elements_local;
  sycl::local_accessor<uint32_t, 1> active_elements_tail;
  sycl::local_accessor<bool, 1> visited;
  sycl::local_accessor<T, 1> work_group_reduce;
  sycl::local_accessor<uint32_t, 1> work_group_reduce_tail;
  LambdaT functor;
};


namespace workgroup_mapped {

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::FrontierType F, typename LambdaT>
sygraph::event launchBitmapKernel(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, F>& in,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, F>& out,
                                  LambdaT&& functor) {
  if constexpr (F != sygraph::frontier::FrontierType::bitmap && F != sygraph::frontier::FrontierType::bitvec) {
    throw std::runtime_error("Invalid frontier type");
  }

  sycl::queue& q = graph.getQueue();

  size_t bitmap_range = in.getBitmapRange();
  size_t num_nodes = graph.getVertexCount();

  auto in_dev_frontier = in.getDeviceFrontier();
  auto out_dev_frontier = out.getDeviceFrontier();
  auto graph_dev = graph.getDeviceGraph();

  using bitmap_kernel_t = bitmap_kernel<T, decltype(in_dev_frontier), decltype(graph_dev), LambdaT>;

  size_t offsets_size = in.computeActiveFrontier();

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * bitmap_range;
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<uint32_t, 1> n_edges_local{local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> active_elements_tail{types::detail::MAX_SUBGROUPS, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> work_group_reduce{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> work_group_reduce_tail{1, cgh};


    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range},
                                                            bitmap_kernel_t{num_nodes,
                                                                            in_dev_frontier,
                                                                            out_dev_frontier,
                                                                            graph_dev,
                                                                            n_edges_local,
                                                                            active_elements_local,
                                                                            active_elements_tail,
                                                                            visited,
                                                                            work_group_reduce,
                                                                            work_group_reduce_tail,
                                                                            std::forward<LambdaT>(functor)});
  });
  return {e};
}

template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::FrontierType FT, typename LambdaT>
sygraph::event launchVectorKernel(GraphT& graph,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FT>& in,
                                  const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, FT>& out,
                                  LambdaT&& functor) {
  sycl::queue& q = graph.getQueue();

  const T* active_elements = in.getVector();
  size_t active_elements_size = in.getVectorSize();

  auto inDevFrontier = in.getDeviceFrontier();
  auto outDevFrontier = out.getDeviceFrontier();
  auto graphDev = graph.getDeviceGraph();

  using vector_kernel_t = vector_kernel<T, decltype(inDevFrontier), decltype(graphDev), LambdaT>;

  auto e = q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{512}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{
        active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    sycl::local_accessor<uint32_t, 1> n_edges_local{local_range, cgh};
    sycl::local_accessor<bool, 1> visited{local_range, cgh};
    sycl::local_accessor<T, 1> active_elements_local{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> ids{local_range, cgh};
    sycl::local_accessor<uint32_t, 1> active_elements_local_tail{types::detail::MAX_SUBGROUPS, cgh};
    sycl::local_accessor<T, 1> pad{8192 /*outDevFrontier.getVectorMaxSize()*/, cgh};
    sycl::local_accessor<uint32_t, 1> pad_tail{1, cgh};

    cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                     vector_kernel_t(active_elements,
                                     active_elements_size,
                                     inDevFrontier,
                                     outDevFrontier,
                                     graphDev,
                                     n_edges_local,
                                     visited,
                                     active_elements_local,
                                     active_elements_local_tail,
                                     ids,
                                     pad,
                                     pad_tail,
                                     std::forward<LambdaT>(functor)));
  });

  return {e};
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitvec>& out,
                      LambdaT&& functor) {
  if (in.useVector()) {
    return launchVectorKernel(graph, in, out, std::forward<LambdaT>(functor));
  } else {
    return launchBitmapKernel(graph, in, out, std::forward<LambdaT>(functor));
  }
}


template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::bitmap>& out,
                      LambdaT&& functor) {
  return launchBitmapKernel(graph, in, out, std::forward<LambdaT>(functor));
}

template<graph::detail::GraphConcept GraphT, typename T, typename LambdaT>
sygraph::event vertex(GraphT& graph,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& in,
                      const sygraph::frontier::Frontier<T, sygraph::frontier::FrontierView::vertex, sygraph::frontier::FrontierType::vector>& out,
                      LambdaT&& functor) {
  return launchVectorKernel(graph, in, out, std::forward<LambdaT>(functor));
}

} // namespace workgroup_mapped
} // namespace detail
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph