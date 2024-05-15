#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/sycl/event.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

namespace workgroup_mapped {

template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event vertex(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  using type_t = typename in_frontier_t::type_t;
  size_t active_elements_size = in.get_num_active_elements();

  type_t* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{64}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<type_t, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<type_t, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};
    
    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      // 0. retrieve global and local ids
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      size_t local_range = item.get_local_range(0);
      auto group = item.get_group();
      auto group_id = item.get_group_linear_id();
      auto subgroup = item.get_sub_group();
      auto subgroup_id = subgroup.get_group_id();
      size_t subgroup_size = subgroup.get_local_range()[0];
      size_t sgid = subgroup.get_local_linear_id();

      // 1. load number of edges in local memory
      if (group.leader()) {
        work_group_reduce_tail[0] = 0;
      }
      if (gid < active_elements_size) {
        type_t element = active_elements[gid];
        n_edges_local[lid] = graphDev.get_neighbors_count(element);
        active_elements_local[lid] = element;
        visited[lid] = false;
      } else {
        n_edges_local[lid] = 0;
        visited[lid] = true;
      }

      // // 1.5 compute nodes to be computed by all the item in the workgroup
      // sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> tail{work_group_reduce_tail[0]};
      // if (n_edges_local[lid] >= local_range && !visited[lid]) {
      //   work_group_reduce[tail++] = lid;
      // }

      // sycl::group_barrier(group); // synchronize

      // // 2. process elements with more than local_range edges
      // for (size_t i = 0; i < tail.load(); i++) { // TODO: [!!!!] for some reason this slows a lot the performances (6ms)
      //   size_t vertex_id = work_group_reduce[i];
      //   auto vertex = active_elements_local[vertex_id];
      //   size_t n_edges = n_edges_local[vertex_id];
      //   size_t private_slice = n_edges / local_range;
      //   auto start = graphDev.begin(vertex) + (private_slice * lid);
      //   auto end = lid == local_range - 1 ? graphDev.end(vertex) : start + private_slice;

      //   for (auto n = start; n != end; ++n) {
      //     auto edge = n.get_index();
      //     auto weight = graphDev.get_edge_weight(edge);
      //     auto neighbor = *n;
      //     if (functor(vertex, neighbor, edge, weight)) {
      //       outDevFrontier.insert(neighbor); // this might be the bottleneck
      //     }
      //   }
      //   if (lid == 0) {
      //     visited[vertex_id] = true;
      //   }
      // }
      sycl::group_barrier(group);

      // 3. process elements with less than local_range edges but more than one subgroup size edges
      for (size_t i = 0; i < subgroup_size; i++) {
        size_t vertex_id = subgroup_id * subgroup_size + i;
        if (!visited[vertex_id] &&  n_edges_local[vertex_id] >= subgroup_size) {
          auto vertex = active_elements_local[vertex_id];
          size_t n_edges = n_edges_local[vertex_id];
          size_t private_slice = n_edges / subgroup_size;
          auto start = graphDev.begin(vertex) + (private_slice * sgid);
          auto end = sgid == subgroup_size - 1 ? graphDev.end(vertex) : start + private_slice;

          for (auto n = start; n != end; ++n) {
            auto edge = n.get_index();
            auto weight = graphDev.get_edge_weight(edge);
            auto neighbor = *n;
            if (functor(vertex, neighbor, edge, weight)) {
              outDevFrontier.insert(neighbor);
            }
          }
          sycl::group_barrier(subgroup);
          if (sgid == i) {
            visited[vertex_id] = true;
          }
        }
      }

      // 4. process the rest
      if (!visited[lid]) {
        auto vertex = active_elements_local[lid];
        auto start = graphDev.begin(vertex);
        auto end = graphDev.end(vertex);

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) {
            outDevFrontier.insert(neighbor);
          }
        }
      }
    });
  });

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }
  return {e};
}

template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event vertex_local_mem(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  using type_t = typename in_frontier_t::type_t;
  size_t active_elements_size = in.get_num_active_elements();

  type_t* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);

  auto e = q.submit([&](sycl::handler& cgh) {

    sycl::range<1> local_range{64}; // TODO: [!] Tune on this value, or compute it dynamically
    sycl::range<1> global_range{active_elements_size > local_range[0] ? active_elements_size + (local_range[0] - (active_elements_size % local_range[0])) : local_range[0]};

    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    sycl::local_accessor<size_t, 1> n_edges_local {local_range, cgh};
    sycl::local_accessor<bool, 1> visited {local_range, cgh};
    sycl::local_accessor<type_t, 1> active_elements_local {local_range, cgh};
    sycl::local_accessor<type_t, 1> work_group_reduce {local_range, cgh};
    sycl::local_accessor<size_t, 1> work_group_reduce_tail {1, cgh};

    constexpr size_t LOCAL_FRONTIER_SIZE = 1024;
    sycl::local_accessor<type_t, 1> l_frontier {local_range[0] * local_range[0], cgh};
    sycl::local_accessor<size_t, 1> l_frontier_tail {1, cgh};
    
    cgh.parallel_for<class workgroup_mapped_advance_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      // 0. retrieve global and local ids
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();
      size_t local_range = item.get_local_range(0);
      auto group = item.get_group();
      auto group_id = item.get_group_linear_id();
      auto subgroup = item.get_sub_group();
      auto subgroup_id = subgroup.get_group_id();
      size_t subgroup_size = subgroup.get_local_range()[0];
      size_t sgid = subgroup.get_local_linear_id();

      sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> refc(l_frontier_tail[0]);

      // 1. load number of edges in local memory
      if (group.leader()) {
        work_group_reduce_tail[0] = 0;
      }
      if (gid < active_elements_size) {
        type_t element = active_elements[gid];
        n_edges_local[lid] = graphDev.get_neighbors_count(element);
        active_elements_local[lid] = element;
        visited[lid] = false;
      } else {
        n_edges_local[lid] = 0;
        visited[lid] = true;
      }

      // // 1.5 compute nodes to be computed by all the item in the workgroup
      // sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> tail{work_group_reduce_tail[0]};
      // if (n_edges_local[lid] >= local_range && !visited[lid]) {
      //   work_group_reduce[tail++] = lid;
      // }

      // sycl::group_barrier(group); // synchronize

      // // 2. process elements with more than local_range edges
      // for (size_t i = 0; i < tail.load(); i++) { // TODO: [!!!!] for some reason this slows a lot the performances (6ms)
      //   size_t vertex_id = work_group_reduce[i];
      //   auto vertex = active_elements_local[vertex_id];
      //   size_t n_edges = n_edges_local[vertex_id];
      //   size_t private_slice = n_edges / local_range;
      //   auto start = graphDev.begin(vertex) + (private_slice * lid);
      //   auto end = lid == local_range - 1 ? graphDev.end(vertex) : start + private_slice;

      //   for (auto n = start; n != end; ++n) {
      //     auto edge = n.get_index();
      //     auto weight = graphDev.get_edge_weight(edge);
      //     auto neighbor = *n;
      //     if (functor(vertex, neighbor, edge, weight)) {
      //       outDevFrontier.insert(neighbor); // this might be the bottleneck
      //     }
      //   }
      //   if (lid == 0) {
      //     visited[vertex_id] = true;
      //   }
      // }
      sycl::group_barrier(group);

      // 3. process elements with less than local_range edges but more than one subgroup size edges
      for (size_t i = 0; i < subgroup_size; i++) {
        size_t vertex_id = subgroup_id * subgroup_size + i;
        if (!visited[vertex_id] &&  n_edges_local[vertex_id] >= subgroup_size) {
          auto vertex = active_elements_local[vertex_id];
          size_t n_edges = n_edges_local[vertex_id];
          size_t private_slice = n_edges / subgroup_size;
          auto start = graphDev.begin(vertex) + (private_slice * sgid);
          auto end = sgid == subgroup_size - 1 ? graphDev.end(vertex) : start + private_slice;

          for (auto n = start; n != end; ++n) {
            auto edge = n.get_index();
            auto weight = graphDev.get_edge_weight(edge);
            auto neighbor = *n;
            if (functor(vertex, neighbor, edge, weight)) {
              if (refc.load() < LOCAL_FRONTIER_SIZE) {
                l_frontier[refc++] = neighbor;
              } else {
                outDevFrontier.insert(neighbor);
              }
            }
          }
          sycl::group_barrier(subgroup);
          if (sgid == i) {
            visited[vertex_id] = true;
          }
        }
      }

      // 4. process the rest
      if (!visited[lid]) {
        auto vertex = active_elements_local[lid];
        auto start = graphDev.begin(vertex);
        auto end = graphDev.end(vertex);

        for (auto n = start; n != end; ++n) {
          auto edge = n.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *n;
          if (functor(vertex, neighbor, edge, weight)) {
            if (refc.load() < LOCAL_FRONTIER_SIZE) {
              l_frontier[refc++] = neighbor;
            } else {
              outDevFrontier.insert(neighbor);
            }
          }
        }
      }

      // 5. copy local frontier to global frontier
      sycl::group_barrier(group);
      size_t address_space = 0;
      if (group.leader()) {
        address_space = outDevFrontier.prealloc(refc.load());
      }
      address_space = sycl::group_broadcast(group, address_space, 0);
      for (size_t i = lid; i < refc.load(); i += item.get_local_range(0)) {
        outDevFrontier.insert(l_frontier[i], address_space + i);
      }
    });
  });

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }
  return {e};
}

} // namespace workitem_mapped
} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph