#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/frontier/frontier_settings.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {
namespace intersection {
namespace detail {

template<graph::detail::GraphConcept GraphT, typename T, typename sygraph::frontier::FrontierView FrontierView, typename LambdaT>
sygraph::event bitmapExecute(GraphT& graph,
                             const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& in1,
                             const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& in2,
                             const sygraph::frontier::Frontier<T, FrontierView, sygraph::frontier::FrontierType::bitmap>& out,
                             LambdaT&& functor) {
  out.clear();
  out.merge(in1).wait();
  return out.intersect(in2);
}

} // namespace detail
} // namespace intersection
} // namespace operators
} // namespace v0
} // namespace sygraph