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

#include <sygraph/operators/for/impl_for.hpp>

namespace sygraph {
namespace operators {

namespace compute {

template<typename T, typename R>
concept ReducerT = std::same_as<T, sycl::plus<R>> || std::same_as<T, sycl::multiplies<R>> || std::same_as<T, sycl::minimum<R>>
                   || std::same_as<T, sycl::maximum<R>>;


/**
 * @brief Executes a given functor over a graph and its frontier.
 *
 * This function launches a bitmap kernel to perform computations on the graph
 * using the provided frontier and functor.
 *
 * @tparam GraphT The type of the graph, which must satisfy the GraphConcept.
 * @tparam T The type of the elements in the frontier.
 * @tparam FrontierType The type of the frontier.
 * @tparam LambdaT The type of the functor to be executed.
 *
 * @param graph The graph on which the computation is to be performed.
 * @param frontier The frontier containing the elements to be processed.
 * @param functor The functor to be executed on the graph and frontier.
 * @param expected_size An optional parameter specifying the expected size of the frontier.
 *
 * @return An Event object representing the execution of the functor.
 */
template<frontier::frontier_view FW, graph::detail::GraphConcept GraphT, typename T, typename LambdaT, frontier::frontier_type FT>
sygraph::Event execute(GraphT& graph,
                       const sygraph::frontier::Frontier<T, FT>& frontier,
                       LambdaT&& functor,
                       frontier::size::frontier_size_t expected_size = frontier::size::fetch_from_memory) {
  return sygraph::operators::compute::detail::launchBitmapKernel<FW>(graph, frontier, std::forward<LambdaT>(functor), expected_size);
}

/**
 * @brief Reduces values over a graph and its frontier using a specified reduction operator.
 *
 * This function launches a kernel to perform reduction operations on the graph
 * using the provided frontier, reduction operator, and accumulator.
 *
 * @tparam FW The frontier view type.
 * @tparam ReductionOperator The type of the reduction operator.
 * @tparam GraphT The type of the graph, which must satisfy the GraphConcept.
 * @tparam T The type of the elements in the frontier.
 * @tparam R The type of the accumulator.
 * @tparam FT The type of the frontier.
 * @tparam LambdaT The type of the function to be executed for reduction.
 *
 * @param graph The graph on which the reduction is to be performed.
 * @param frontier The frontier containing the elements to be processed.
 * @param accumulator The accumulator to store the reduced result. Its initial value takes part in the reduction, and
 *                    it holds the result when the function returns.
 * @param function The function to be executed for reduction.
 * @param expected_size An optional parameter specifying the expected size of the frontier.
 *
 * @return An Event object representing the execution of the reduction operation.
 */
template<frontier::frontier_view FW,
         typename ReductionOperator,
         graph::detail::GraphConcept GraphT,
         typename T,
         typename R,
         frontier::frontier_type FT,
         typename LambdaT>
  requires ReducerT<ReductionOperator, R>
sygraph::Event reduce(GraphT& graph,
                      const sygraph::frontier::Frontier<T, FT>& frontier,
                      R& accumulator,
                      LambdaT&& function,
                      frontier::size::frontier_size_t expected_size = frontier::size::fetch_from_memory) {
  return sygraph::operators::compute::detail::launchBitmapReduce<FW, ReductionOperator>(
      graph, frontier, accumulator, std::forward<LambdaT>(function), expected_size);
}

} // namespace compute
} // namespace operators
} // namespace sygraph