#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::triangle_3);

  sygraph::algorithms::TC tc(graph);
  tc.init();
  tc.run();

  assert(tc.getNumTriangles() == 1);

  // Complete graph on 4 vertices: every vertex triple is a triangle.
  auto k4 = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::complete_4);
  sygraph::algorithms::TC tc_k4(k4);
  tc_k4.init();
  tc_k4.run();
  assert(tc_k4.getNumTriangles() == 4);
}
