/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#include <sygraph/sync/atomics.hpp>

#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>

namespace sygraph {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct BCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;
  /// Path counts, dependencies and centrality are fractional, so they are not stored in an integral weight type.
  using centrality_t = std::conditional_t<std::is_floating_point_v<weight_t>, weight_t, float>;

  const vertex_t invalid = std::numeric_limits<vertex_t>::max();

  GraphType& G;
  vertex_t source;

  vertex_t* labels;
  centrality_t* deltas;
  centrality_t* sigmas;
  centrality_t* bc_values;

  BCInstance(GraphType& G, const vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    labels = sygraph::memory::detail::memoryAlloc<vertex_t, memory::space::device>(size, queue);
    deltas = sygraph::memory::detail::memoryAlloc<centrality_t, memory::space::device>(size, queue);
    sigmas = sygraph::memory::detail::memoryAlloc<centrality_t, memory::space::device>(size, queue);
    bc_values = sygraph::memory::detail::memoryAlloc<centrality_t, memory::space::device>(size, queue);

    queue.fill(labels, static_cast<vertex_t>(this->invalid), size);
    queue.fill(deltas, static_cast<centrality_t>(0), size);
    queue.fill(sigmas, static_cast<centrality_t>(0), size);
    queue.fill(bc_values, static_cast<centrality_t>(0), size);
    queue.wait_and_throw();

    queue.fill(&sigmas[source], static_cast<centrality_t>(1), 1);
    queue.fill(&labels[source], static_cast<vertex_t>(0), 1);
    queue.wait_and_throw();
  }

  ~BCInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(labels, queue);
    memory::detail::releaseUSM(deltas, queue);
    memory::detail::releaseUSM(sigmas, queue);
    memory::detail::releaseUSM(bc_values, queue);
  }
};
} // namespace detail

/**
 * @class BC
 * @brief A class template for computing Betweenness Centrality on a graph.
 *
 * The BC class template provides methods to initialize, reset, and run the Betweenness Centrality algorithm on a
 * given graph. It uses SYCL for parallel execution and supports profiling.
 *
 * @tparam GraphType The type of the graph on which the BC algorithm will be executed.
 */
template<typename GraphType>
class BC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  using centrality_t = typename detail::BCInstance<GraphType>::centrality_t;

  BC(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the BCInstance with the given source vertex.
   *
   * This function creates a new instance of BCInstance for the provided graph
   * and source vertex, and assigns it to the internal _instance member.
   *
   * @param source The source vertex from which to initialize the BCInstance.
   */
  void init(const vertex_t source) { _instance = std::make_unique<detail::BCInstance<GraphType>>(_g, source); }

  /**
   * @brief Resets the internal state of the instance.
   *
   * This function calls the reset method on the internal instance,
   * effectively resetting its state to the initial configuration.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Executes the Betweenness Centrality (BC) algorithm.
   *
   * This function runs the BC algorithm on the graph instance. It initializes the necessary
   * frontiers and iteratively processes the graph in two phases: forward and backward.
   *
   * In the forward phase, it propagates the shortest path labels and sigma values from the source
   * vertex to all other vertices. In the backward phase, it computes the delta values and updates
   * the BC values for each vertex.
   *
   * @throws std::runtime_error if the BC instance is not initialized.
   */
  void run() {
    if (!_instance) { throw std::runtime_error("BC instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();
    auto source = _instance->source;

    auto in_frontier = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_type::mlb>(queue, G);

    in_frontier.insert(source);

    vertex_t invalid = _instance->invalid;
    vertex_t* labels = _instance->labels;
    centrality_t* deltas = _instance->deltas;
    centrality_t* sigmas = _instance->sigmas;
    centrality_t* bc_values = _instance->bc_values;

    using frontier_state_t = typename decltype(in_frontier)::frontier_state_type;
    std::vector<frontier_state_t> frontiers_states;

    while (!in_frontier.empty()) {
      auto e = sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                     sygraph::frontier::frontier_view::vertex,
                                                     sygraph::frontier::frontier_view::vertex>(
          G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            vertex_t new_label = labels[src] + 1;
            vertex_t old_label = invalid;
            sygraph::sync::cas(&labels[dst], old_label, new_label);

            if (old_label != invalid && old_label != new_label) { return false; }

            sygraph::sync::atomicFetchAdd(sigmas + dst, sigmas[src]);
            return old_label == invalid;
          });
      e.wait_and_throw();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "BC::Forward");
#endif
      _depth++;
      _search_depth++;

      frontiers_states.push_back(out_frontier.saveState());
      sygraph::frontier::swap(out_frontier, in_frontier);
      out_frontier.clear();
    }

    while (_depth > 0) {
      in_frontier.loadState(frontiers_states.back());
      frontiers_states.pop_back();

      auto e = sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                     sygraph::frontier::frontier_view::vertex,
                                                     sygraph::frontier::frontier_view::none>(
          G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            if (src == source) { return false; }

            auto s_label = labels[src];
            auto d_label = labels[dst];
            if (s_label + 1 != d_label) { return false; }

            auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
            sygraph::sync::atomicFetchAdd(deltas + src, update);
            sygraph::sync::atomicFetchAdd(bc_values + src, update);

            return false;
          });
      e.wait_and_throw();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "BC::Backward");
#endif
      _depth--;
      _search_depth++;
      if (isBackwardConverged()) { break; }
    }

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;
  }

  /**
   * @brief Returns the betweenness centrality computed by the last run.
   *
   * The value of a vertex v is the dependency of the source s on v: the sum, over every target t, of the fraction of
   * shortest s-t paths that pass through v. The source and the vertices it cannot reach have centrality 0.
   *
   * @return A vector with the centrality of each vertex.
   */
  std::vector<centrality_t> getCentrality() const {
    if (!_instance) { throw std::runtime_error("BC instance not initialized"); }
    std::vector<centrality_t> values(_instance->G.getVertexCount());
    _instance->G.getQueue().copy(_instance->bc_values, values.data(), values.size()).wait();
    return values;
  }

protected:
  bool isBackwardConverged() {
    if (_depth == 0) {
      _backward = false;
      return true;
    }

    return false;
  }

  virtual bool isConverged() { return !_forward && !_backward; }

  bool _forward = true;
  bool _backward = true;

  size_t _depth = 0;
  size_t _search_depth = 1;
  const size_t _max_depth = 5;

private:
  GraphType& _g;
  std::unique_ptr<sygraph::algorithms::detail::BCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
