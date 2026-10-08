#include "test_utils.hpp"

template<typename GraphT>
void expectLabels(GraphT& graph, const std::vector<uint>& expected, uint source) {
  sygraph::algorithms::CC cc(graph);
  cc.init(source);
  cc.run();
  assert(cc.getLabels() == expected);
}

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::io::storage::matrices::two_cc);

  // Each vertex is labelled with the largest vertex id of its component, whatever the source.
  sygraph::algorithms::CC cc(graph);
  cc.init(0);
  cc.run();
  sygraph::tests::expectEqual(cc.getLabels(), std::array<uint, 6>{4, 4, 4, 4, 4, 5});
  cc.reset();
  cc.init(5);
  cc.run();
  sygraph::tests::expectEqual(cc.getLabels(), std::array<uint, 6>{4, 4, 4, 4, 4, 5});

  // Many components and isolated vertices; repeated because concurrent label updates race.
  constexpr size_t n = 10000;
  const auto csr = sygraph::tests::gen::componentsGraph(n, 7, 50, n, /*seed=*/53);
  const auto expected = sygraph::tests::ref::ccMaxLabel(csr);
  auto random_graph = sygraph::tests::buildGraph(q, csr);
  for (int repeat = 0; repeat < 5; ++repeat) { expectLabels(random_graph, expected, 0); }
  auto device_graph = sygraph::tests::buildGraph<sygraph::memory::space::device>(q, csr);
  expectLabels(device_graph, expected, n - 1);
}
