#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  sygraph::graph::Properties properties;
  properties.directed = true;
  properties.weighted = true;
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::weighted_directed_5, properties);

  sygraph::algorithms::SSSP sssp(graph);
  uint source = 0;
  sssp.init(source);
  sssp.run();

  std::vector<uint> distances(graph.getVertexCount());
  for (size_t i = 0; i < distances.size(); ++i) { distances[i] = sssp.getDistance(i); }

  sygraph::tests::expectEqual(distances, std::array<uint, 5>{0, 1, 3, 4, 5});
  sygraph::tests::expectEqual(sssp.getDistances(), std::array<uint, 5>{0, 1, 3, 4, 5});

  // From another source, vertices 0 and 1 cannot be reached.
  constexpr uint unreachable = std::numeric_limits<uint>::max();
  static_assert(decltype(sssp)::unreachable() == unreachable);
  sssp.init(2);
  sssp.run();
  sygraph::tests::expectEqual(sssp.getDistances(), std::array<uint, 5>{unreachable, unreachable, 0, 1, 2});

  // Shortest distances larger than the vertex count.
  auto heavy = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::heavy_line_3, {.directed = false, .weighted = true});
  sygraph::algorithms::SSSP sssp_heavy(heavy);
  sssp_heavy.init(0);
  sssp_heavy.run();
  sygraph::tests::expectEqual(sssp_heavy.getDistances(), std::array<uint, 3>{0, 10, 20});
  sssp_heavy.init(1);
  sssp_heavy.run();
  sygraph::tests::expectEqual(sssp_heavy.getDistances(), std::array<uint, 3>{10, 0, 10});
}
