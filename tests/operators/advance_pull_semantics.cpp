#include "test_utils.hpp"

#include <vector>

using sygraph::frontier::frontier_type;
using sygraph::frontier::frontier_view;
using sygraph::operators::direction;
using sygraph::operators::load_balancer;
namespace size = sygraph::frontier::size;
using sygraph::tests::bitsWhere;
using sygraph::tests::insertWhere;
using sygraph::tests::readBits;
using sygraph::tests::gen::csr_t;

// A pull advance processes every vertex v that is NOT in the input frontier, over the edges of the inverse graph
// (v <- u), visiting only neighbours u that ARE in the frontier. The functor is called as (v, u, edge, weight) and a
// `true` result inserts v into the output frontier.
struct Calls {
  sycl::queue& q;
  uint32_t* count;
  uint32_t* src;
  uint32_t* dst;

  Calls(sycl::queue& q, size_t edges) : q(q) {
    count = sycl::malloc_shared<uint32_t>(edges, q);
    src = sycl::malloc_shared<uint32_t>(edges, q);
    dst = sycl::malloc_shared<uint32_t>(edges, q);
    q.fill(count, 0U, edges).wait();
  }
  ~Calls() {
    sycl::free(count, q);
    sycl::free(src, q);
    sycl::free(dst, q);
  }
};

enum class functor_kind {
  modulo,      // true when (v + u) % 3 == 0
  always,      // always true
  largest_only // true only for the largest in-neighbour of v that is in the frontier
};

template<direction D, load_balancer Lb, typename GraphT, typename PredT>
void runPull(sycl::queue& q,
             GraphT& graph,
             const csr_t& inverse,
             PredT pred,
             functor_kind kind,
             size::frontier_size_t expected_size = size::fetch_from_memory) {
  const size_t n = graph.getVertexCount();
  const auto active = bitsWhere(n, pred);
  const auto& offsets = inverse.getRowOffsets();
  const auto& columns = inverse.getColumnIndices();

  // Largest in-neighbour of every vertex that is in the frontier (n when there is none).
  std::vector<uint32_t> largest(n, static_cast<uint32_t>(n));
  for (size_t v = 0; v < n; ++v) {
    for (size_t e = offsets[v]; e < offsets[v + 1]; ++e) {
      if (active[columns[e]] && (largest[v] == n || columns[e] > largest[v])) { largest[v] = columns[e]; }
    }
  }
  uint32_t* largest_dev = sycl::malloc_shared<uint32_t>(n, q);
  std::copy(largest.begin(), largest.end(), largest_dev);

  auto in = sygraph::frontier::makeFrontier<frontier_view::vertex, frontier_type::mlb>(q, graph);
  auto out = sygraph::frontier::makeFrontier<frontier_view::vertex, frontier_type::mlb>(q, graph);
  insertWhere(q, in, pred);

  Calls calls{q, inverse.getNumNonzeros()};
  auto* count = calls.count;
  auto* src_seen = calls.src;
  auto* dst_seen = calls.dst;
  sygraph::operators::advance::frontier<D, Lb, frontier_view::vertex, frontier_view::vertex>(
      graph,
      in,
      out,
      [=](auto v, auto u, auto edge, auto) -> bool {
        sygraph::sync::atomicFetchAdd(count + edge, 1U);
        src_seen[edge] = v;
        dst_seen[edge] = u;
        switch (kind) {
          case functor_kind::modulo: return (v + u) % 3 == 0;
          case functor_kind::always: return true;
          case functor_kind::largest_only: return u == largest_dev[v];
        }
        return false;
      },
      expected_size)
      .waitAndThrow();

  std::vector<bool> expected_output(n, false);
  for (size_t v = 0; v < n; ++v) {
    uint32_t calls_v = 0;
    uint32_t candidates = 0;
    for (size_t e = offsets[v]; e < offsets[v + 1]; ++e) {
      const uint32_t u = columns[e];
      const bool candidate = !active[v] && active[u];
      candidates += candidate;
      calls_v += calls.count[e];
      if (!candidate) {
        assert(calls.count[e] == 0);
        continue;
      }
      assert(calls.count[e] <= 1);
      if (calls.count[e] > 0) {
        assert(calls.src[e] == v);
        assert(calls.dst[e] == u);
      }
      const bool accepted = kind == functor_kind::always || (kind == functor_kind::modulo && (v + u) % 3 == 0)
                            || (kind == functor_kind::largest_only && u == largest[v]);
      if (accepted) { expected_output[v] = true; }
      if constexpr (D == direction::pull_all) { assert(calls.count[e] == 1); }
    }
    if constexpr (D == direction::pull) {
      // The short-circuit is best effort: lanes working on the same vertex in parallel can each call the functor, but
      // a vertex with candidates is visited at least once and no edge more than once.
      if (candidates > 0) { assert(calls_v >= 1 && calls_v <= candidates); }
    }
  }
  assert(readBits(q, out) == expected_output);
  sycl::free(largest_dev, q);
}

template<load_balancer Lb, typename GraphT>
void runAll(sycl::queue& q, GraphT& graph, const csr_t& inverse) {
  const size_t n = graph.getVertexCount();
  const size_t range = sygraph::types::detail::byte_size * sizeof(sygraph::types::bitmap_type_t);
  auto pattern = [=](size_t v) { return (v * 2654435761u) % 3 == 0 || v < 2 * range || v == n - 1; };

  for (auto expected_size : {size::fetch_from_memory, size::infer_from_device, 1, static_cast<int>(3 * n)}) {
    runPull<direction::pull_all, Lb>(q, graph, inverse, pattern, functor_kind::modulo, expected_size);
  }
  runPull<direction::pull, Lb>(q, graph, inverse, pattern, functor_kind::always);
  runPull<direction::pull, Lb>(q, graph, inverse, pattern, functor_kind::largest_only);
  runPull<direction::pull, Lb>(q, graph, inverse, pattern, functor_kind::modulo, size::infer_from_device);
  // Every vertex active: nothing to pull.
  runPull<direction::pull_all, Lb>(q, graph, inverse, [](size_t) { return true; }, functor_kind::always);
  // No vertex active: every vertex is processed, but none has a neighbour in the frontier.
  runPull<direction::pull_all, Lb>(q, graph, inverse, [](size_t) { return false; }, functor_kind::always);
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 10000;

  for (bool directed : {false, true}) {
    const auto csr = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/directed ? 5 : 13, directed);
    // Pull walks the inverse graph; for undirected graphs it is the graph itself.
    const auto inverse = directed ? csr.invert() : csr;
    auto graph = sygraph::tests::buildGraph(q, csr, {.directed = directed});
    runAll<load_balancer::workgroup_mapped>(q, graph, inverse);
    runAll<load_balancer::bucketing>(q, graph, inverse);
  }

  const auto csr = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/17, /*directed=*/true);
  auto device_graph = sygraph::tests::buildGraph<sygraph::memory::space::device>(q, csr, {.directed = true});
  runAll<load_balancer::workgroup_mapped>(q, device_graph, csr.invert());
}
