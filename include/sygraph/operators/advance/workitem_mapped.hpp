#pragma once

#include <sycl/sycl.hpp>
#include <memory>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

namespace detail {

namespace workitem_mapped {

template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event vertex(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  using type_t = typename in_frontier_t::type_t;
  
  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  type_t* active_elements;
  if (!in.self_allocated()) {
    active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  }
  in.get_active_elements(active_elements, active_elements_size);
  
  // TODO: [!!] for some reason is way slower so we need to investigate
  // constexpr size_t THRESHOLD = 1; // TODO: we must tune on a certain value to avoid offloading computation when the frontier is too small
  // if (active_elements_size <= THRESHOLD) {
  //   for (size_t k = 0; k < active_elements_size; k++) {
  //     auto deviceGraph = graph.get_device_graph();
  //     auto element = active_elements[k];
  //     auto start = deviceGraph.begin(element);
  //     auto end = deviceGraph.end(element);

  //     for (auto i = start; i != end; ++i) {
  //       auto edge = i.get_index();
  //       auto weight = graph.get_edge_weight(edge);
  //       auto neighbor = *i;
  //       if (functor(element, neighbor, edge, weight)) {
  //         out.insert(neighbor);
  //       }
  //     }
  //   }
  //   return sygraph::event{};
  // }

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();
    sycl::stream os(1024, 256, cgh); // TODO remove

    cgh.parallel_for<class vertex_workitem_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.get_index();
        auto weight = graphDev.get_edge_weight(edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) {
          bool val = outDevFrontier.insert(neighbor);
          if (!val) {
            os << "Error inserting " << neighbor << sycl::endl;
          }
        }
      }
    });
  })};

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }
  return ret;
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
  in.get_active_elements(active_elements, active_elements_size);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();
    constexpr size_t LOCAL_MEM_SIZE = types::detail::MAX_LOCAL_MEM_SIZE;

    sycl::range<1> local_size(128); // TODO: tune this value
    sycl::range<1> global_size(active_elements_size <= local_size[0] ? local_size[0] : (active_elements_size + local_size[0] - active_elements_size % local_size[0]));

    auto local_context = in.get_local_context(cgh);
    using local_context_type = decltype(local_context);

    sycl::stream os {1024, 64, cgh};

    cgh.parallel_for<class vertex_local_mem_advance_kernel>(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
      size_t gid = item.get_global_linear_id();
      size_t lid = item.get_local_linear_id();

      local_context.init(item);

      sycl::group_barrier(item.get_group());

      if (gid < active_elements_size) {
        auto element = active_elements[gid];
        auto start = graphDev.begin(element);
        auto end = graphDev.end(element);

        // each work item takes care of all the neighbors of the vertex he is responsible for
        for (auto i = start; i != end; ++i) {
          auto edge = i.get_index();
          auto weight = graphDev.get_edge_weight(edge);
          auto neighbor = *i;
          if (functor(element, neighbor, edge, weight)) {
            if (!local_context.insert(neighbor)) {
              // outDevFrontier.insert(neighbor);
            }
            outDevFrontier.insert(neighbor); // TODO remove
          }
        }
      }
      // if (lid == 0) {
      //   os << "Group: " << item.get_group(0) << " | " << local_context.tail[0] << sycl::endl;
      // }
      sycl::group_barrier(item.get_group());
      size_t prealloc_size = local_context.get_prealloc_size(item);
      size_t address_space = 0;
      if (lid == 0) {
        os << "Group: " << item.get_group(0) << " | " << prealloc_size << sycl::endl;
        // address_space = outDevFrontier.prealloc(prealloc_size);
      }
      address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
      // for (size_t i = lid; i < prealloc_size; i += item.get_local_range(0)) {
      //   outDevFrontier.insert(local_context.data[i], prealloc_size + i);
      // }
      // local_context.copy_to_global(item, outDevFrontier);
    });
  })};
  std::cout << std::endl;

  if (!in.self_allocated()) {
    sycl::free(active_elements, q);
  }

  return ret;
}
// template <typename graph_t,
//           typename in_frontier_t,
//           typename out_frontier_t,
//           typename lambda_t>
// sygraph::event vertex_local_mem(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
//   sycl::queue& q = graph.get_queue();

//   using type_t = typename in_frontier_t::type_t;
//   size_t active_elements_size = in.get_num_active_elements();
//   type_t* active_elements;
//   in.get_active_elements(active_elements, active_elements_size);

//   sygraph::event ret {q.submit([&](sycl::handler& cgh) {
//     auto inDevFrontier = in.get_device_frontier();
//     auto outDevFrontier = out.get_device_frontier();
//     auto graphDev = graph.get_device_graph();
//     constexpr size_t LOCAL_MEM_SIZE = types::detail::MAX_LOCAL_MEM_SIZE;

//     sycl::range<1> local_size(128); // TODO: tune this value
//     sycl::range<1> global_size(active_elements_size <= local_size[0] ? local_size[0] : (active_elements_size + local_size[0] - active_elements_size % local_size[0]));

//     sycl::local_accessor<type_t, 1> l_frontier(LOCAL_MEM_SIZE, cgh);
//     sycl::local_accessor<size_t, 1> l_frontier_tail(1, cgh);

//     cgh.parallel_for<class vertex_local_mem_advance_kernel>(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
//       size_t gid = item.get_global_linear_id();
//       size_t lid = item.get_local_linear_id();
//       sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> l_frontier_tail_ref(l_frontier_tail[0]);

//       if (lid == 0) {
//         l_frontier_tail_ref = 0;
//       }
//       sycl::group_barrier(item.get_group());

//       if (gid < active_elements_size) {
//         auto element = active_elements[gid];
//         auto start = graphDev.begin(element);
//         auto end = graphDev.end(element);

//         // each work item takes care of all the neighbors of the vertex he is responsible for
//         for (auto i = start; i != end; ++i) {
//           auto edge = i.get_index();
//           auto weight = graphDev.get_edge_weight(edge);
//           auto neighbor = *i;
//           if (functor(element, neighbor, edge, weight)) {
//             if (l_frontier_tail_ref < LOCAL_MEM_SIZE) { // if the local memory is not full, we can use it
//               l_frontier[l_frontier_tail_ref++] = neighbor;
//             } else { // if the local memory is full, we need to use the global mem
//               outDevFrontier.insert(neighbor);
//             }
//           }
//         }
//       }
//       sycl::group_barrier(item.get_group());
//       size_t address_space = 0;
//       if (lid == 0) {
//         address_space = outDevFrontier.prealloc(l_frontier_tail_ref.load());
//       }
//       address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
//       for (size_t i = lid; i < l_frontier_tail_ref; i += item.get_local_range(0)) {
//         outDevFrontier.insert(l_frontier[i], address_space + i);
//       }
//     });
//   })};

//   if (!in.self_allocated()) {
//     sycl::free(active_elements, q);
//   }

//   return ret;
// }

template <typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event edge(graph_t& graph, const in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  
  sycl::queue& q = graph.get_queue();

  using type_t = typename in_frontier_t::type_t;
  size_t active_elements_size = in.get_num_active_elements();
  type_t* active_elements = sycl::malloc_shared<type_t>(active_elements_size, q);
  in.get_active_elements(active_elements);

  sygraph::event ret {q.submit([&](sycl::handler& cgh) {
    auto inDevFrontier = in.get_device_frontier();
    auto outDevFrontier = out.get_device_frontier();
    auto graphDev = graph.get_device_graph();

    cgh.parallel_for<class edge_advance_kernel>(sycl::range<1>(active_elements_size), [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      auto start = graphDev.begin(element);
      auto end = graphDev.end(element);

      // each work item takes care of all the neighbors of the vertex he is responsible for
      for (auto i = start; i != end; ++i) {
        auto edge = i.get_index();
        auto weight = graphDev.get_edge_weight(element, edge);
        auto neighbor = *i;
        if (functor(element, neighbor, edge, weight)) {
          outDevFrontier.insert(edge);
        }
      }
    });
  })};

  sycl::free(active_elements, q);
  return ret;
}

} // namespace workitem_mapped
} // namespace detail  
} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph