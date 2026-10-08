#include "test_utils.hpp"

int main() {
  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);

  sygraph::algorithms::BFS bfs_push(graph);
  uint source = 0;
  bfs_push.init(source);
  auto push_details = bfs_push.run(sygraph::algorithms::bfs_direction::push);
  sygraph::tests::expectEqual(bfs_push.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
  assert(!push_details.push_steps.empty());
  assert(push_details.pull_steps.empty());
  bfs_push.reset();

  sygraph::algorithms::BFS bfs_pull(graph);
  bfs_pull.init(source);
  auto pull_details = bfs_pull.run(sygraph::algorithms::bfs_direction::pull);
  sygraph::tests::expectEqual(bfs_pull.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
  assert(pull_details.push_steps.empty());
  assert(!pull_details.pull_steps.empty());
  bfs_pull.reset();

  sygraph::algorithms::BFS bfs_hybrid(graph);
  bfs_hybrid.init(source);
  auto hybrid_details = bfs_hybrid.run(sygraph::algorithms::bfs_direction::hybrid, 1.0f, 1.0f);
  sygraph::tests::expectEqual(bfs_hybrid.getDistances(), std::array<uint, 5>{0, 1, 2, 3, 4});
  assert(hybrid_details.iterations == 5);

  // Unreachable vertices keep distance n + 1; a source can be passed as a literal.
  auto two_cc = sygraph::tests::buildGraphFromMatrix(q, sygraph::io::storage::matrices::two_cc);
  sygraph::algorithms::BFS bfs_two_cc(two_cc);
  bfs_two_cc.init(0);
  bfs_two_cc.run();
  sygraph::tests::expectEqual(bfs_two_cc.getDistances(), std::array<uint, 6>{0, 1, 1, 2, 2, 7});
  assert(bfs_two_cc.getDistance(5) == 7);
  assert(bfs_two_cc.getDistance(3) == 2);
  bfs_two_cc.init(5);
  bfs_two_cc.run();
  sygraph::tests::expectEqual(bfs_two_cc.getDistances(), std::array<uint, 6>{7, 7, 7, 7, 7, 0});

  // Parents: one level closer to the source, -1 for the source and unreached vertices.
  bfs_two_cc.init(3);
  bfs_two_cc.run();
  sygraph::tests::expectEqual(bfs_two_cc.getDistances(), std::array<uint, 6>{2, 2, 1, 0, 2, 7});
  const auto parents = bfs_two_cc.getParents();
  const uint none = static_cast<uint>(-1);
  sygraph::tests::expectEqual(parents, std::array<uint, 6>{2, 2, 3, none, 2, none});
  assert(bfs_two_cc.getParent(2) == 3);

  // Hybrid on a star: one push step reaches every leaf, then nothing is left unexplored, so it switches to pull.
  constexpr size_t leaves = 99;
  std::vector<sygraph::tests::gen::Edge> spokes;
  for (uint v = 1; v <= leaves; ++v) { spokes.push_back({0, v, 1}); }
  auto star = sygraph::tests::buildGraph(q, sygraph::tests::gen::csrFromEdges(leaves + 1, spokes, /*directed=*/false));
  sygraph::algorithms::BFS bfs_star(star);
  bfs_star.init(0);
  auto star_details = bfs_star.run(sygraph::algorithms::bfs_direction::hybrid, 1.0f, 1.0f);
  assert(star_details.push_steps == std::set<size_t>{0});
  assert(star_details.pull_steps == std::set<size_t>{1});
  assert(star_details.iterations == 2);
  std::vector<uint> star_distances(leaves + 1, 1);
  star_distances[0] = 0;
  sygraph::tests::expectEqual(bfs_star.getDistances(), star_distances);
}
