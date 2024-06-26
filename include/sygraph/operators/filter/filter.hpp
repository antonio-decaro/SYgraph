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

#include <sygraph/operators/filter/bitmap_filter_impl.hpp>
#include <sygraph/operators/filter/vector_filter_impl.hpp>

namespace sygraph {
inline namespace v0 {
namespace operators {

namespace filter {

template<graph::detail::GraphConcept GraphT,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename LambdaT>
sygraph::event inplace(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& frontier, LambdaT&& functor) {
  return sygraph::operators::filter::detail::inplace(graph, frontier, std::forward<LambdaT>(functor));
}

template<typename GraphT,
         typename T,
         typename sygraph::frontier::FrontierView FrontierView,
         typename sygraph::frontier::FrontierType FrontierType,
         typename LambdaT>
sygraph::event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& in,
                        const sygraph::frontier::Frontier<T, FrontierView, FrontierType>& out,
                        LambdaT&& functor) {
  return sygraph::operators::filter::detail::external(graph, in, out, std::forward<LambdaT>(functor));
}


} // namespace filter
} // namespace operators
} // namespace v0
} // namespace sygraph