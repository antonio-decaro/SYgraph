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
inline namespace v0 {
namespace operators {

namespace compute {

namespace detail {

template<typename graph_t, typename T, sygraph::frontier::FrontierView FW, sygraph::frontier::FrontierType FT, typename lambda_t>
sygraph::event launchBitmapKernel(graph_t& graph, const sygraph::frontier::Frontier<T, FW, FT>& frontier, lambda_t&& functor) {
  if constexpr (FT != sygraph::frontier::FrontierType::bitmap && FT != sygraph::frontier::FrontierType::bitvec) {
    throw std::runtime_error("Invalid frontier type");
  }
  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  size_t num_nodes = graph.getVertexCount();

  size_t bitmap_range = frontier.getBitmapRange();
  size_t offsets_size = frontier.computeActiveFrontier();

  return q.submit([&](sycl::handler& cgh) {
    sycl::range<1> local_range{bitmap_range};
    size_t global_size = offsets_size * local_range[0];
    sycl::range<1> global_range{global_size > local_range[0] ? global_size + (local_range[0] - (global_size % local_range[0])) : local_range[0]};

    cgh.parallel_for<class for_kernel>(sycl::nd_range<1>{global_range, local_range}, [=](sycl::nd_item<1> item) {
      auto lid = item.get_local_id();
      auto group_id = item.get_group_linear_id();
      auto local_size = item.get_local_range()[0];
      int* bitmap_offsets = dev_frontier.getOffsets();

      size_t actual_id = bitmap_offsets[group_id] * bitmap_range + lid;

      if (actual_id < num_nodes && dev_frontier.check(actual_id)) { functor(actual_id); }
    });
  });
}

template<typename graph_t, typename T, typename sygraph::frontier::FrontierView FrontierView, typename lambda_t>
sygraph::event
execute(graph_t& graph, const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& frontier, lambda_t&& functor) {
  return launchBitmapKernel(graph, frontier, functor);
}

template<typename graph_t, typename T, typename sygraph::frontier::FrontierView FrontierView, typename lambda_t>
sygraph::event
execute(graph_t& graph, const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::vector>& frontier, lambda_t&& functor) {
  auto q = graph.getQueue();

  size_t active_elements_size = types::detail::MAX_ACTIVE_ELEMS_SIZE;
  T* active_elements;
  if (!frontier.selfAllocated()) { active_elements = memory::detail::memoryAlloc<T, memory::space::shared>(active_elements_size, q); }
  frontier.getActiveElements(active_elements, active_elements_size);

  sygraph::event e = q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class for_kernel>(sycl::range<1>{active_elements_size}, [=](sycl::id<1> idx) {
      auto element = active_elements[idx];
      functor(element);
    });
  });

  if (!frontier.selfAllocated()) { sycl::free(active_elements, q); }

  return e;
}

template<typename graph_t, typename T, typename sygraph::frontier::FrontierView FrontierView, typename lambda_t>
sygraph::event
execute(graph_t& graph, const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitvec>& frontier, lambda_t&& functor) {
  sygraph::event e;
  auto q = graph.getQueue();
  auto dev_frontier = frontier.getDeviceFrontier();

  if (dev_frontier.useVector()) {
    T* active_elements = dev_frontier.getVector();
    size_t size = dev_frontier.getVectorSize();

    e = q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>{size}, [=](sycl::id<1> idx) {
        auto element = active_elements[idx];
        functor(element);
      });
    });
  } else {
    e = launchBitmapKernel(graph, frontier, functor);
  }

  return e;
}

} // namespace detail
} // namespace compute
} // namespace operators
} // namespace v0
} // namespace sygraph