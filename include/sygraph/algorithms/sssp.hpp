/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

#include <limits>
#include <memory>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/filter/filter.hpp>
#include <sygraph/operators/for/for.hpp>
#include <vector>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <sygraph/sync/atomics.hpp>


namespace sygraph {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct SSSPInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  /// Distance of the vertices that cannot be reached from the source.
  static constexpr weight_t unreachable = std::numeric_limits<weight_t>::max();

  GraphType& G;
  vertex_t source;
  weight_t* distances;
  vertex_t* parents;
  int* visited;

  SSSPInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    distances = memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);
    queue.fill(distances, unreachable, size).wait();
    queue.fill(&distances[source], static_cast<weight_t>(0), 1).wait();

    parents = memory::detail::memoryAlloc<vertex_t, memory::space::device>(size, queue);
    queue.fill(parents, static_cast<vertex_t>(-1), size).wait();

    visited = memory::detail::memoryAlloc<int, memory::space::device>(size, queue);
    queue.fill(visited, -1, size).wait();
  }

  size_t getVisitedVertices() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_nodes = 0;
    auto& queue = G.getQueue();
    std::vector<weight_t> dists(vertex_count);
    queue.copy(distances, dists.data(), vertex_count).wait();

    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (dists[i] != unreachable) { visited_nodes++; }
    }
    return visited_nodes;
  }

  size_t getVisitedEdges() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_edges = 0;
    auto& queue = G.getQueue();
    std::vector<weight_t> dists(vertex_count);
    queue.copy(distances, dists.data(), vertex_count).wait();

    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (dists[i] != unreachable) { visited_edges += G.getDegree(i); }
    }
    return visited_edges;
  }

  ~SSSPInstance() {
    sycl::queue& queue = G.getQueue();
    memory::detail::releaseUSM(distances, queue);
    memory::detail::releaseUSM(parents, queue);
    memory::detail::releaseUSM(visited, queue);
  }
};
} // namespace detail


/**
 * @class SSSP
 * @brief Single Source Shortest Path (SSSP) algorithm implementation for a given graph type.
 *
 * This class provides methods to initialize, run, and retrieve results of the SSSP algorithm.
 *
 * @tparam GraphType The type of the graph on which the SSSP algorithm will be executed.
 */
template<typename GraphType>
class SSSP {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  /**
   * @brief Constructor to initialize the SSSP algorithm with a given graph.
   *
   * @param g Reference to the graph on which the SSSP algorithm will be executed.
   */
  SSSP(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the Single Source Shortest Path (SSSP) algorithm instance.
   *
   * This function initializes the SSSP algorithm by creating a new instance of
   * the SSSPInstance class with the provided graph and source vertex. It also
   * sets the distance to the source vertex to zero.
   *
   * @param source The source vertex from which to start the SSSP algorithm.
   */
  void init(vertex_t source) { _instance = std::make_unique<detail::SSSPInstance<GraphType>>(_g, source); }


  /**
   * @brief Resets the internal state of the instance.
   *
   * This function calls the reset method on the internal instance,
   * effectively resetting its state to the initial configuration.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Executes the Single Source Shortest Path (SSSP) algorithm.
   *
   * This function runs the SSSP algorithm on a graph instance. It initializes the necessary
   * data structures and iteratively processes the graph to compute the shortest paths from
   * the source vertex to all other vertices.
   *
   * @throws std::runtime_error if the SSSP instance is not initialized.
   *
   * The function performs the following steps:
   * 1. Initializes the in_frontier with the source vertex.
   * 2. Iteratively processes the graph until the in_frontier is empty:
   *    a. Advances the frontier by exploring neighboring vertices and updating distances.
   *    b. Filters the out_frontier to remove already visited vertices.
   *    c. Clears the out_frontier and increments the iteration counter.
   *
   * Profiling events are recorded if ENABLE_PROFILING is defined.
   */
  template<bool EnableProfiling = false>
  void run() {
    if (!_instance) { throw std::runtime_error("SSSP instance not initialized"); }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;
    auto& visited = _instance->visited;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);

    size_t size = G.getVertexCount();

    int iter = 0;
    in_frontier.insert(source);

    while (!in_frontier.empty()) {
      auto e1 = sygraph::operators::advance::
          frontier<load_balance_t::workgroup_mapped, sygraph::frontier::frontier_view::vertex, sygraph::frontier::frontier_view::vertex>(
              G, in_frontier, out_frontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
                weight_t source_distance = sygraph::sync::load(&distances[src]);
                weight_t distance_to_neighbor = source_distance + weight;

                // Check if the destination node has been claimed as someone's child
                weight_t recover_distance = sygraph::sync::load(&distances[dst]);
                recover_distance = sygraph::sync::min(&(distances[dst]), &distance_to_neighbor);

                return (distance_to_neighbor < recover_distance);
              });
      e1.wait();

      auto e2 = sygraph::operators::filter::external(G, out_frontier, in_frontier, [=](auto vertex) -> bool {
        if (visited[vertex] == iter) return false;
        visited[vertex] = iter;
        return true;
      });
      e2.wait();

#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e1, "advance");
      sygraph::Profiler::addEvent(e2, "filter");
#endif

      out_frontier.clear();
      iter++;
    }
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addVisitedEdges(_instance->getVisitedEdges());
#endif
  }

  /**
   * @brief Returns the distance from the source to a vertex, or `unreachable()` if the vertex cannot be reached.
   */
  weight_t getDistance(size_t vertex) const {
    weight_t distance;
    _instance->G.getQueue().copy(_instance->distances + vertex, &distance, 1).wait();
    return distance;
  }

  /**
   * @brief Returns the distances from the source to all vertices; unreachable vertices hold `unreachable()`.
   */
  std::vector<weight_t> getDistances() const {
    std::vector<weight_t> distances(_instance->G.getVertexCount());
    _instance->G.getQueue().copy(_instance->distances, distances.data(), distances.size()).wait();
    return distances;
  }

  /**
   * @brief The distance reported for vertices that cannot be reached from the source.
   */
  static constexpr weight_t unreachable() { return detail::SSSPInstance<GraphType>::unreachable; }

  vertex_t getParents(size_t vertex) const {
    throw std::runtime_error("Not implemented");
    return 0;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::SSSPInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace sygraph
