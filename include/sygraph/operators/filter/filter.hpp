/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
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

namespace sygraph {
namespace operators {

namespace filter {

/**
 * @brief Applies an in-place filter operation on the given graph using the specified functor.
 *
 * This function launches a bitmap kernel to perform the filter operation in-place on the graph.
 * Every element of the frontier for which `functor(element)` returns true is REMOVED from the frontier; the others are
 * kept. Note that this is the opposite of external(), which keeps the elements for which the functor returns true.
 *
 * @tparam GraphT The type of the graph, which must satisfy the GraphConcept.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 * @tparam LambdaT The type of the functor to be applied.
 *
 * @param graph The graph on which the filter operation is to be performed.
 * @param frontier The frontier containing the elements to be filtered.
 * @param functor The functor to be applied for the filter operation; returning true removes the element.
 *
 * @return An Event object representing the status of the filter operation.
 */
template<graph::detail::GraphConcept GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event inplace(GraphT& graph, const sygraph::frontier::Frontier<T, FrontierType>& frontier, LambdaT&& functor) {
  return sygraph::operators::filter::detail::launchBitmapKernelInplace(graph, frontier, functor);
}

/**
 * @brief Applies a filter operation on the input frontier and stores the result in the output frontier.
 *
 * This function uses an external bitmap to perform the filtering operation on the given graph.
 * The output frontier is cleared first, then every element of the input frontier for which `functor(element)` returns
 * true is inserted into it. The input frontier is not modified.
 *
 * @tparam GraphT The type of the graph.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 * @tparam LambdaT The type of the functor used for filtering.
 *
 * @param graph The graph on which the filter operation is performed.
 * @param in The input frontier containing elements to be filtered.
 * @param out The output frontier where the filtered elements will be stored.
 * @param functor The functor used to determine if an element should be included in the output frontier.
 *
 * @return An event representing the status of the filter operation.
 */
template<typename GraphT, typename T, sygraph::frontier::frontier_type FrontierType, typename LambdaT>
sygraph::Event external(GraphT& graph,
                        const sygraph::frontier::Frontier<T, FrontierType>& in,
                        sygraph::frontier::Frontier<T, FrontierType>& out,
                        LambdaT&& functor) {
  return sygraph::operators::filter::detail::launchBitmapKernelExternal(graph, in, out, functor);
}


} // namespace filter
} // namespace operators
} // namespace sygraph