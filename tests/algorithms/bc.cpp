#include "test_utils.hpp"

template<typename GraphT>
std::vector<float> centrality(GraphT& graph, uint source) {
  sygraph::algorithms::BC bc(graph);
  bc.init(source);
  bc.run();
  return bc.getCentrality();
}

int main() {
  auto q = sygraph::tests::makeQueue();
  using sygraph::tests::expectNear;

  // Path 0-1-2-3-4: from 0, vertex k lies on the paths to the 4 - k vertices after it.
  auto line = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);
  sygraph::algorithms::BC bc(line);
  bc.init(0);
  bc.run();
  expectNear(bc.getCentrality(), {0, 3, 2, 1, 0});
  // The same object can be re-initialized with another source.
  bc.init(2);
  bc.run();
  expectNear(bc.getCentrality(), {0, 1, 0, 1, 0});

  // Two shortest paths from 0 to 3: each middle vertex carries half of them.
  auto diamond = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::diamond_4);
  expectNear(centrality(diamond, 0), {0, 0.5, 0.5, 0});

  // From a leaf of a star, the centre lies on the paths to the 3 other leaves.
  auto star = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::star_5);
  expectNear(centrality(star, 1), {3, 0, 0, 0, 0});

  // An isolated source reaches nothing.
  auto two_cc = sygraph::tests::buildGraphFromMatrix(q, sygraph::io::storage::matrices::two_cc);
  expectNear(centrality(two_cc, 5), {0, 0, 0, 0, 0, 0});

  constexpr size_t n = 3000;
  const auto undirected = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/59, /*directed=*/false);
  auto undirected_graph = sygraph::tests::buildGraph(q, undirected);
  for (uint source : {0u, 1234u}) { expectNear(centrality(undirected_graph, source), sygraph::tests::ref::brandes(undirected, source)); }

  const auto directed = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/61, /*directed=*/true);
  auto directed_graph = sygraph::tests::buildGraph(q, directed, {.directed = true});
  expectNear(centrality(directed_graph, 7), sygraph::tests::ref::brandes(directed, 7));
}
