/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sygraph/operators/config.hpp"
#include <sycl/sycl.hpp>

#include <sygraph/frontier/frontier.hpp>
#include <sygraph/graph/graph.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif
#include <memory>
#include <set>

/**
 * @namespace sygraph
 * @brief Namespace for the SYgraph library.
 *
 * The sygraph namespace contains classes and functions for graph algorithms and data structures.
 */
namespace sygraph {
namespace algorithms {

enum class bfs_direction { push, pull, hybrid };

struct BFSRunDetails {
  size_t iterations = 0;
  std::set<size_t> push_steps;
  std::set<size_t> pull_steps;
};

namespace detail {
/**
 * @brief Represents an instance of the Breadth-First Search (BFS) algorithm on a graph.
 *
 * The BFSInstance struct encapsulates the necessary data and operations for performing the BFS algorithm on a graph.
 * It stores the graph, the source vertex, and arrays for distances and parents.
 */
template<typename GraphType>
struct BFSInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

  GraphType& G;      /**< The graph on which the BFS algorithm will be performed. */
  vertex_t source;   /**< The source vertex for the BFS algorithm. */
  edge_t* distances; /**< Array to store the distances from the source vertex to each vertex in the graph. */
  vertex_t* parents; /**< Array to store the parent vertex of each vertex in the graph during the BFS traversal. */

  /**
   * @brief Constructs a BFSInstance object.
   *
   * @param G The graph on which the BFS algorithm will be performed.
   * @param source The source vertex for the BFS algorithm.
   */
  BFSInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    // Initialize distances
    distances = memory::detail::memoryAlloc<edge_t, memory::space::device>(size, queue);
    parents = memory::detail::memoryAlloc<vertex_t, memory::space::device>(size, queue);
    queue.fill(distances, static_cast<edge_t>(size + 1), size).wait_and_throw();
    queue.fill(&distances[source], static_cast<edge_t>(0), 1);
    // Initialize parents
    queue.fill(parents, static_cast<vertex_t>(-1), size);
    queue.wait_and_throw();

  }

  size_t getVisitedVertices() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_nodes = 0;
    auto& queue = G.getQueue();
    std::vector<edge_t> dists (vertex_count);
    queue.copy(distances, dists.data(), vertex_count).wait();

    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (dists[i] != static_cast<edge_t>(vertex_count + 1)) { visited_nodes++; }
    }
    return visited_nodes;
  }

  size_t getVisitedEdges() const {
    size_t vertex_count = G.getVertexCount();
    size_t visited_edges = 0;
    auto& queue = G.getQueue();
    std::vector<edge_t> dists (vertex_count); 
    queue.copy(distances, dists.data(), vertex_count).wait();
    
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      if (dists[i] != static_cast<edge_t>(vertex_count + 1)) { visited_edges += G.getDegree(i); }
    }
    return visited_edges;
  }

  /**
   * @brief Destroys the BFSInstance object and frees the allocated memory.
   */
  ~BFSInstance() {
    memory::detail::releaseUSM(distances, G.getQueue());
    memory::detail::releaseUSM(parents, G.getQueue());
  }
};
} // namespace detail

/**
 * @brief Represents the Breadth-First Search (BFS) algorithm.
 *
 * The BFS algorithm is used to traverse or search a graph in a breadthward motion,
 * starting from a given source vertex. It visits all the vertices at the same level
 * before moving to the next level.
 *
 * @tparam GraphType The type of the graph on which the BFS algorithm will be performed.
 */
template<typename GraphType>
class BFS {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

public:
  /**
   * @brief Constructs a BFS object.
   */
  BFS(GraphType& g) : _g(g) {};

  /**
   * @brief Initializes the BFS algorithm with the given graph and source vertex.
   *
   * @param G The graph on which the BFS algorithm will be performed.
   * @param source The source vertex for the BFS algorithm.
   */
  void init(vertex_t source) { _instance = std::make_unique<detail::BFSInstance<GraphType>>(_g, source); }

  /**
   * @brief Resets the BFS algorithm.
   */
  void reset() { _instance.reset(); }

  /**
   * @brief Runs the BFS algorithm.
   *
   * @param direction The direction of the BFS traversal (push, pull, or hybrid).
   * @param alpha The alpha parameter for the hybrid BFS heuristic. Used to switch from push to pull.
   * @param beta The beta parameter for the hybrid BFS heuristic. Used to switch from pull to push.
   * @tparam EnableProfiling A boolean flag to enable profiling.
   * @throws std::runtime_error if the BFS instance is not initialized.
   */
  BFSRunDetails run(bfs_direction direction = bfs_direction::push, float alpha = 1.0f, float beta = 1.0f) {
    BFSRunDetails details;
    if (!_instance) { throw std::runtime_error("BFS instance not initialized"); }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;

    sycl::queue& queue = G.getQueue();

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;

    auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);
    auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(queue, G);

    in_frontier.insert(source);

    size_t size = G.getVertexCount();
    auto g_device = G.getDeviceGraph();
    int iter = 0;

    sygraph::Event e;

    auto push_step = [&]() {
      return sygraph::operators::advance::frontier<load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
          G,
          in_frontier,
          out_frontier,
          [=](auto src, auto dst, auto edge, auto weight) -> bool {
            if (distances[dst] == size + 1) {
              distances[dst] = iter + 1;
              // Several sources may race here; any of them is a valid parent (all are at distance iter).
              parents[dst] = src;
              return true;
            }
            return false;
          },
          sygraph::frontier::size::fetch_from_memory);
    };

    auto pull_step = [&]() {
      return sygraph::operators::advance::
          frontier<direction_t::pull, load_balance_t::workgroup_mapped, frontier_view_t::vertex, frontier_view_t::vertex>(
              G,
              in_frontier,
              out_frontier,
              [=](auto src, auto dst, auto edge, auto weight) -> bool {
                if (distances[src] == size + 1 && distances[dst] == iter) {
                  distances[src] = iter + 1;
                  parents[src] = dst;
                  return true;
                }
                return false;
              },
              sygraph::frontier::size::fetch_from_memory);
    };

    bool push = direction != bfs_direction::pull;
    size_t n_push_step = 0;
    size_t n_pull_step = 0;

    while (!in_frontier.empty()) {
      if (push) {
        e = push_step();
        details.push_steps.insert(iter);
      } else {
        e = pull_step();
        details.pull_steps.insert(iter);
      }
      e.waitAndThrow();
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "advance");
#endif
      evaluateHeuristic(alpha, beta, out_frontier, direction, push);
      sygraph::frontier::swap(in_frontier, out_frontier);
      out_frontier.clear();
      iter++;
    }

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addVisitedEdges(_instance->getVisitedEdges());
#endif
    details.iterations = iter;
    return details;
  }

  /**
   * @brief Returns the distances from the source vertex to a vertex in the graph.
   *
   * @param vertex The vertex for which to get the distance.
   * @return A pointer to the array of distances.
   */
  edge_t getDistance(size_t vertex) const {
    edge_t distance;
    _instance->G.getQueue().copy(_instance->distances + vertex, &distance, 1).wait();
    return distance;
  }

  /**
   * @brief Returns the distances from the source vertex to all vertices in the graph.
   *
   * @return A vector of distances.
   */
  std::vector<edge_t> getDistances() const {
    std::vector<edge_t> distances(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    queue.copy(_instance->distances, distances.data(), distances.size()).wait();
    return distances;
  }

  /**
   * @brief Returns the parent vertices for a vertex in the graph.
   *
   * @param vertex The vertex for which to get the parent vertices.
   * @return A pointer to the array of parent vertices.
   */
  vertex_t getParent(size_t vertex) const {
    vertex_t parent;
    _instance->G.getQueue().copy(_instance->parents + vertex, &parent, 1).wait();
    return parent;
  }

  /**
   * @brief Returns the parent vertices for all vertices in the graph.
   *
   * The parent of a reached vertex is a neighbour one level closer to the source. The source and the unreached
   * vertices have parent `vertex_t(-1)`.
   *
   * @return A vector of parent vertices.
   */
  std::vector<vertex_t> getParents() const {
    std::vector<vertex_t> parents(_instance->G.getVertexCount());
    sycl::queue& queue = _instance->G.getQueue();
    queue.copy(_instance->parents, parents.data(), parents.size()).wait();
    return parents;
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::BFSInstance<GraphType>> _instance;

  template<typename F>
  size_t fetchFrontierDegree(const F& frontier) {
    size_t total_degree = 0;
    auto& G = _instance->G;
    auto g_device = G.getDeviceGraph();

    auto e = sygraph::operators::compute::reduce<sygraph::frontier::frontier_view::vertex, sycl::plus<size_t>>(
        G, frontier, total_degree, [=](auto v, auto& accumulator) { accumulator += g_device.getDegree(v); });
    e.waitAndThrow();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeFrontierDegree");
#endif
    return total_degree;
  }

  uint32_t fetchUnexploredDegree() {
    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    auto g_device = G.getDeviceGraph();

    sycl::buffer<uint32_t, 1> degree_buf(sycl::range<1>(1));

    auto distances = _instance->distances;
    size_t nodes = G.getVertexCount();

    auto e = queue.submit([&](sycl::handler& cgh) {
      auto sum_reduction = sycl::reduction<uint32_t>(degree_buf, cgh, sycl::plus<uint32_t>(), sycl::property::reduction::initialize_to_identity{});

      cgh.parallel_for(sycl::range<1>(nodes), sum_reduction, [=](sycl::id<1> idx, auto& sum) {
        size_t vertex = idx[0];
        if (distances[vertex] == nodes + 1) { sum += static_cast<uint32_t>(g_device.getDegree(vertex)); }
      });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeUnexploredDegree");
#endif
    auto degree_host = degree_buf.get_host_access();
    return degree_host[0];
  }

  template<typename F>
  bool shouldSwitchToPull(const F& frontier, float alpha) {
    uint32_t frontier_degree = fetchFrontierDegree(frontier);
    uint32_t unexplored_degree = fetchUnexploredDegree();

    return frontier_degree > (unexplored_degree / alpha);
  }

  template<typename F>
  bool shouldSwitchToPush(const F& frontier, float beta) {
    auto& G = _instance->G;
    auto frontier_size = frontier.size();
    auto total_nodes = G.getVertexCount();
    return frontier_size < (total_nodes / beta);
  }

  template<typename F>
  void evaluateHeuristic(const float alpha, const float beta, const F& frontier, bfs_direction direction, bool& push) {
    if (direction == bfs_direction::hybrid) {
      if (push) {
        if (shouldSwitchToPull(frontier, alpha)) { push = false; }
      } else {
        if (shouldSwitchToPush(frontier, beta)) { push = true; }
      }
    }
  }
};

} // namespace algorithms
} // namespace sygraph
