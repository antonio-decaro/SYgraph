#include "test_utils.hpp"

template<typename GraphT>
size_t countTriangles(GraphT& graph) {
  sygraph::algorithms::TC tc(graph);
  tc.init();
  tc.run();
  return tc.getNumTriangles();
}

int main() {
  auto q = sygraph::tests::makeQueue();

  auto triangle = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::triangle_3);
  assert(countTriangles(triangle) == 1);

  // Complete graph on 4 vertices: every vertex triple is a triangle.
  auto k4 = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::complete_4);
  assert(countTriangles(k4) == 4);

  // No triangles in a path.
  auto line = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);
  assert(countTriangles(line) == 0);

  // C(40, 3) triangles.
  auto k40 = sygraph::tests::buildGraph(q, sygraph::tests::gen::completeGraph(40));
  assert(countTriangles(k40) == 9880);

  const auto random = sygraph::tests::gen::randomGraph(2000, 20000, /*seed=*/67, /*directed=*/false);
  auto random_graph = sygraph::tests::buildGraph(q, random);
  assert(countTriangles(random_graph) == sygraph::tests::ref::triangles(random));

  const auto components = sygraph::tests::gen::componentsGraph(2000, 5, 20, 4000, /*seed=*/71);
  auto components_graph = sygraph::tests::buildGraph(q, components);
  assert(countTriangles(components_graph) == sygraph::tests::ref::triangles(components));
}
