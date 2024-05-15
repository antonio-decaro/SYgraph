#include <sycl/sycl.hpp>

#include <sygraph/graph/graph.hpp>
#include <sygraph/frontier/frontier.hpp>
#include <sygraph/operators/advance/advance.hpp>
#include <sygraph/operators/for/for.hpp>
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
inline namespace v0 {
namespace algorithms {
namespace detail {
/**
 * @brief Represents an instance of the Breadth-First Search (BFS) algorithm on a graph.
 *
 * The BFSInstance struct encapsulates the necessary data and operations for performing the BFS algorithm on a graph.
 * It stores the graph, the source vertex, and arrays for distances and parents.
 */
template <typename GraphType>
struct BFSInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;

  GraphType& G; /**< The graph on which the BFS algorithm will be performed. */
  vertex_t source; /**< The source vertex for the BFS algorithm. */
  edge_t* distances; /**< Array to store the distances from the source vertex to each vertex in the graph. */
  vertex_t* parents; /**< Array to store the parent vertex of each vertex in the graph during the BFS traversal. */

  /**
   * @brief Constructs a BFSInstance object.
   *
   * @param G The graph on which the BFS algorithm will be performed.
   * @param source The source vertex for the BFS algorithm.
   */
  BFSInstance(GraphType& G, vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.get_queue();
    size_t size = G.get_vertex_count();

    // Initialize distances
    distances = sycl::malloc_shared<edge_t>(size, queue);
    queue.fill(distances, static_cast<edge_t>(size + 1), size).wait();
    distances[source] = static_cast<edge_t>(0); // Distance from source to itself is 0

    // Initialize parents
    parents = sycl::malloc_shared<vertex_t>(size, queue);
    queue.fill(parents, static_cast<vertex_t>(-1), size).wait();
  }

  /**
   * @brief Destroys the BFSInstance object and frees the allocated memory.
   */
  ~BFSInstance() {
    sycl::free(distances, G.get_queue());
    sycl::free(parents, G.get_queue());
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
template<typename GraphType> // TODO: Implement the get_parents method. 
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
  void init(vertex_t& source) {
    _instance = std::make_unique<detail::BFSInstance<GraphType>>(_g, source);
  }

  /**
   * @brief Resets the BFS algorithm.
   */
  void reset() {
    _instance.reset();
  }

  /**
   * @brief Runs the BFS algorithm.
   *
   * @tparam enable_profiling A boolean flag to enable profiling.
   * @throws std::runtime_error if the BFS instance is not initialized.
   */
  template <bool enable_profiling = false> 
  void run() {
    if (!_instance) {
      throw std::runtime_error("BFS instance not initialized");
    }

    auto& G = _instance->G;
    auto& source = _instance->source;
    auto& distances = _instance->distances;
    auto& parents = _instance->parents;

    sycl::queue& queue = G.get_queue();

    using load_balance_t = sygraph::operators::LoadBalancer;
    using direction_t = sygraph::operators::Direction;
    using frontier_view_t = sygraph::frontier::FrontierView;
    using frontier_impl_t = sygraph::frontier::FrontierType;

    auto inFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::vector>(queue, G);
    auto outFrontier = sygraph::frontier::make_frontier<frontier_view_t::vertex, frontier_impl_t::vector>(queue, G);

    inFrontier.insert(source);

    size_t size = G.get_vertex_count();
    int iter = 0;

    // TODO: Add automatic load_balancing for the type of graph. 
    while (!inFrontier.empty()) {
      auto e1 = sygraph::operators::advance::vertex<load_balance_t::workgroup_mapped>(G, inFrontier, outFrontier, [=](auto src, auto dst, auto edge, auto weight) -> bool {
        return (iter + 1) < distances[dst];
      });
      e1.wait_and_throw();
      auto e2 = sygraph::operators::parallel_for::execute(G, outFrontier, [=](auto v) {
        distances[v] = iter + 1;
      });
      e2.wait_and_throw();
      
#ifdef ENABLE_PROFILING
      sygraph::profiler::add_event(e1, "advance");
      sygraph::profiler::add_event(e2, "for");
#endif

      sygraph::frontier::swap(inFrontier, outFrontier);
      outFrontier.clear();
      iter++;
    }
  }

  /**
   * @brief Returns the distances from the source vertex to a vertex in the graph.
   *
   * @param vertex The vertex for which to get the distance.
   * @return A pointer to the array of distances.
   */
  const edge_t get_distance(size_t vertex) const {
    return _instance->distances[vertex];
  }

  /**
   * @brief Returns the parent vertices for a vertex in the graph.
   *
   * @param vertex The vertex for which to get the parent vertices.
   * @return A pointer to the array of parent vertices.
   */
  const vertex_t get_parents(size_t vertex) const {
    throw std::runtime_error("Not implemented");
    return _instance->parents[vertex];
  }

private:
  GraphType& _g;
  std::unique_ptr<detail::BFSInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph