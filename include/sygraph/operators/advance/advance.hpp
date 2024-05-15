#pragma once

#include <memory>
#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/operators/advance/workitem_mapped.hpp>
#include <sygraph/operators/advance/workgroup_mapped.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace advance {

template <sygraph::operators::LoadBalancer lb,
          typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event vertex(graph_t& graph, in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::vertex_local_mem(graph, in, out, std::forward<lambda_t>(functor));
    // return sygraph::operators::advance::detail::vertex_local_mem(graph, in, out, std::forward<lambda_t>(functor));
  } else if (lb == sygraph::operators::LoadBalancer::workgroup_mapped) {
    return sygraph::operators::advance::detail::workgroup_mapped::vertex(graph, in, out, std::forward<lambda_t>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

template <sygraph::operators::LoadBalancer lb,
          typename graph_t,
          typename in_frontier_t,
          typename out_frontier_t,
          typename lambda_t>
sygraph::event edge(graph_t& graph, in_frontier_t& in, out_frontier_t& out, lambda_t&& functor) {
  if constexpr (lb == sygraph::operators::LoadBalancer::workitem_mapped) {
    return sygraph::operators::advance::detail::workitem_mapped::edge(graph, in, out, std::forward<lambda_t>(functor));
  } else {
    throw std::runtime_error("Load balancer not implemented");
  }
}

} // namespace advance
} // namespace operators
} // namespace v0
} // namespace sygraph