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
inline namespace v0 {
namespace algorithms {
namespace detail {

template<typename GraphType>
struct BCInstance {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

  GraphType& G;
  vertex_t source;

  vertex_t* labels;
  weight_t* deltas;
  weight_t* sigmas;
  weight_t* bc_values;

  BCInstance(GraphType& G, const vertex_t source) : G(G), source(source) {
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();

    labels = sygraph::memory::detail::memoryAlloc<vertex_t, memory::space::device>(size, queue);
    deltas = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);
    sigmas = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);
    bc_values = sygraph::memory::detail::memoryAlloc<weight_t, memory::space::device>(size, queue);

    queue.fill(labels, size + 1, size);
    queue.fill(deltas, 0, size);
    queue.fill(sigmas, 0, size);
    queue.fill(bc_values, 0, size);
    queue.wait_and_throw();

    queue.fill(sigmas + source, 1, 1);
    queue.fill(labels + source, 0, 1);
    queue.wait_and_throw();
  }

  ~BCInstance() {
    sycl::queue& queue = G.getQueue();
    sycl::free(labels, queue);
    sycl::free(deltas, queue);
    sycl::free(sigmas, queue);
    sycl::free(bc_values, queue);
  }
};
} // namespace detail

template<typename GraphType>
class BC {
  using vertex_t = typename GraphType::vertex_t;
  using edge_t = typename GraphType::edge_t;
  using weight_t = typename GraphType::weight_t;

public:
  BC(GraphType& g) : _g(g){};

  void init(const vertex_t source) { _instance = std::make_unique<detail::BCInstance<GraphType>>(_g, source); }

  void reset() { _instance.reset(); }

  void run() {
    if (!_instance) { throw std::runtime_error("BC instance not initialized"); }

    auto& G = _instance->G;
    sycl::queue& queue = G.getQueue();
    size_t size = G.getVertexCount();
    auto source = _instance->source;

    auto frontier = sygraph::frontier::makeFrontier<frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);
    std::vector<decltype(frontier)> frontiers;
    for (int i = 1; i < _max_depth; i++) {
      frontiers[i] = sygraph::frontier::makeFrontier<frontier::frontier_view::vertex, sygraph::frontier::frontier_type::hierachic_bitmap>(queue, G);
    }

    auto is_converged = [&](size_t depth) {
      auto& f = frontiers[depth];
      return f.empty();
    };

    frontiers[0] = frontier;
    frontiers[0].insert(_instance->source);

    vertex_t invalid = size + 1;
    vertex_t* labels = _instance->labels;
    weight_t* deltas = _instance->deltas;
    weight_t* sigmas = _instance->sigmas;
    weight_t* bc_values = _instance->bc_values;

    while (!isConverged()) {
      if (_forward) {
        auto op = [=](auto src, auto dst, auto edge, auto weight) -> bool {
          auto new_label = labels[src] + 1;
          auto old_label = sygraph::sync::cas(labels + dst, invalid, new_label);

          if (old_label != invalid && new_label != old_label) { return false; }

          sygraph::sync::atomicFetchAdd(sigmas + dst, sigmas[src]);
          return old_label == invalid;
        };

        while (true) {
          auto in_frontier = frontiers[_depth];
          auto out_frontier = frontiers[_depth + 1];

          sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                sygraph::frontier::frontier_view::vertex,
                                                sygraph::frontier::frontier_view::vertex>(
              G, in_frontier, out_frontier, std::forward<decltype(op)>(op));
          _depth++;
          _search_depth++;
          if (is_converged(_depth)) { break; }
        }
      } else if (_backward) {
        _forward = false;
        auto op = [=](auto src, auto dst, auto edge, auto weight) -> bool {
          if (src == source) { return false; }

          auto s_label = labels[src];
          auto d_label = labels[dst];
          if (s_label + 1 != d_label) { return false; }

          auto update = sigmas[src] / sigmas[dst] * (1 + deltas[dst]);
          sygraph::sync::atomicFetchAdd(deltas + src, update);
          sygraph::sync::atomicFetchAdd(bc_values + src, update);

          return false;
        };

        while (true) {
          auto in_frontier = frontiers[_depth];
          auto out_frontier = frontiers[_depth + 1];

          sygraph::operators::advance::frontier<sygraph::operators::load_balancer::workgroup_mapped,
                                                sygraph::frontier::frontier_view::vertex,
                                                sygraph::frontier::frontier_view::none>(G, in_frontier, out_frontier, std::forward<decltype(op)>(op));
          _depth--;
          _search_depth++;
          if (isBackwardConverged()) { break; }
        }
      }
    }

    using load_balance_t = sygraph::operators::load_balancer;
    using direction_t = sygraph::operators::direction;
    using frontier_view_t = sygraph::frontier::frontier_view;
    using frontier_impl_t = sygraph::frontier::frontier_type;
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
  const size_t _max_depth = 1000;

private:
  GraphType& _g;
  std::unique_ptr<sygraph::algorithms::detail::BCInstance<GraphType>> _instance;
};

} // namespace algorithms
} // namespace v0
} // namespace sygraph
