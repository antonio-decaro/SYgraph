#include "test_utils.hpp"

#include <algorithm>
#include <vector>

using sygraph::algorithms::bfs_direction;
using sygraph::tests::gen::csr_t;

// Every reached vertex other than the source must have a parent one level closer with an edge parent -> vertex.
void expectValidParents(const csr_t& csr, const std::vector<uint>& dist, const std::vector<uint>& parents, uint source) {
  const size_t n = dist.size();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  const uint none = static_cast<uint>(-1);
  for (uint v = 0; v < n; ++v) {
    if (v == source || dist[v] == n + 1) {
      assert(parents[v] == none);
      continue;
    }
    const uint p = parents[v];
    assert(p < n);
    assert(dist[p] + 1 == dist[v]);
    assert(std::binary_search(columns.begin() + offsets[p], columns.begin() + offsets[p + 1], v));
  }
}

template<typename GraphT>
void runAll(sycl::queue& q, GraphT& graph, const csr_t& csr, std::initializer_list<uint> sources) {
  struct Mode {
    bfs_direction direction;
    float alpha;
    float beta;
  };
  for (uint source : sources) {
    const auto expected = sygraph::tests::ref::bfs(csr, source);
    for (auto mode : {Mode{bfs_direction::push, 1, 1},
                      Mode{bfs_direction::pull, 1, 1},
                      Mode{bfs_direction::hybrid, 1, 1},
                      Mode{bfs_direction::hybrid, 0.05f, 20}}) {
      sygraph::algorithms::BFS bfs(graph);
      bfs.init(source);
      bfs.run(mode.direction, mode.alpha, mode.beta);
      const auto distances = bfs.getDistances();
      assert(distances == expected);
      expectValidParents(csr, distances, bfs.getParents(), source);
    }
  }
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 10000;

  // Sparse random graphs leave some vertices unreachable.
  const auto undirected = sygraph::tests::gen::randomGraph(n, 3 * n, /*seed=*/31, /*directed=*/false);
  auto undirected_graph = sygraph::tests::buildGraph(q, undirected);
  runAll(q, undirected_graph, undirected, {0, n / 2});

  const auto directed = sygraph::tests::gen::randomGraph(n, 3 * n, /*seed=*/37, /*directed=*/true);
  auto directed_graph = sygraph::tests::buildGraph(q, directed, {.directed = true});
  runAll(q, directed_graph, directed, {0, n / 2});

  // Several components and isolated vertices (vertex 1 is isolated).
  const auto components = sygraph::tests::gen::componentsGraph(n, 7, 50, n, /*seed=*/41);
  auto components_graph = sygraph::tests::buildGraph(q, components);
  runAll(q, components_graph, components, {0, 1});

  auto device_graph = sygraph::tests::buildGraph<sygraph::memory::space::device>(q, directed, {.directed = true});
  runAll(q, device_graph, directed, {n / 2});
}
