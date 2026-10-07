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

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph {
namespace algorithms {
namespace detail {
/**
 * @brief Represents an instance of the Breadth-First Search (CC) algorithm on a graph.
 *
 * The CCInstance struct encapsulates the necessary data and operations for performing the CC algorithm on a graph.
 * It stores the graph, the source vertex, and arrays for distances and parents.
 */
template<typename GraphType>
struct CCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

  GraphType& G;     /**< The graph on which the CC algorithm will be performed. */
  vertex_t source;  /**< The source vertex for the CC algorithm. */
  vertex_t* labels; /**< Array to store the labels of each vertex in the graph during the CC traversal. */

  /**
   * @brief Constructs a CCInstance object.
   *
   * @param G The graph on which the CC algorithm will be performed.
   * @param source The source vertex for the CC algorithm.
   */
  CCInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    labels = memory::detail::memoryAlloc<vertex_t, memory::space::device>(size, queue);
    queue.fill(labels, static_cast<vertex_t>(-1), size).wait();
  }

  /**
   * @brief Destroys the CCInstance object and frees the allocated memory.
   */
  ~CCInstance() { memory::detail::releaseUSM(labels, G.getQueue()); }
};
} // namespace detail

/**
 * @brief Represents the Breadth-First Search (CC) algorithm.
 *
 * The CC algorithm is used to traverse or search a graph in a breadthward motion,
 * starting from a given source vertex. It visits all the vertices at the same level
 * before moving to the next level.
 *
 * @tparam GraphType The type of the graph on which the CC algorithm will be performed.
 */
template<typename GraphType> // TODO: Implement the getParents method.
class CC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

public:
  /**
   * @brief Constructs a CC object.
   */
  CC(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the CC algorithm with the given graph and source vertex.
   *
   * @param G The graph on which the CC algorithm will be performed.
   * @param source The source vertex for the CC algorithm.
   */
  void init(vertex_t source) { _instance = std::make_unique<detail::CCInstance<GraphType>>(_g, source); }

  /**
   * @brief Resets the CC algorithm.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Runs the CC algorithm.
   *
   * @tparam EnableProfiling A boolean flag to enable profiling.
   * @throws std::runtime_error if the CC instance is not initialized.
   */
  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("CC instance not initialized"); }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& labels = _instance->labels;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);

    queue
        .submit([&](sycl::handler& cgh) {
          cgh.parallel_for(sycl::range<1>(G.getVertexCount()), [=](sycl::item<1> item) {
            vertex_t i = item.get_id();
            labels[i] = i;
          });
        })
        .wait();

    in_frontier.insert(source);

    size_t size = G.getVertexCount();
    int iter = 0;

    auto e1 = sygraph::operators::advance::vertices<load_balance_t::workgroup_mapped, frontier_view_t::vertex>(
        G, in_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
          // Atomic max: a plain load/compare/store lets a smaller label overwrite a larger one.
          vertex_t src_label = sygraph::sync::load(&labels[src]);
          return sygraph::sync::max(&labels[dst], src_label) < src_label;
        });
    e1.waitAndThrow();

    // TODO: Add automatic load_balancing for the type of graph.
    while (!in_frontier.empty()) {
      auto e1 = sygraph::operators::advance::frontier<load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
          G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
            vertex_t src_label = sygraph::sync::load(&labels[src]);
            return sygraph::sync::max(&labels[dst], src_label) < src_label;
          });
      e1.waitAndThrow();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e1, "advance");
#endif

      sygraph::frontier::swap(in_frontier, out_frontier);
      out_frontier.clear();
      iter++;
    }
  }

  /**
   * @brief Returns the component label of every vertex.
   *
   * Vertices in the same connected component share a label: the largest vertex id of the component.
   *
   * @return A vector with the label of each vertex.
   */
  std::vector<vertex_t> getLabels() const {
    if (!_instance) { throw std::runtime_error("CC instance not initialized"); }
    std::vector<vertex_t> labels(_instance->G.getVertexCount());
    _instance->G.getQueue().copy(_instance->labels, labels.data(), labels.size()).wait();
    return labels;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::CCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
