#include "test_utils.hpp"

int main() {
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_type_t = sygraph::frontier::frontier_type;

  auto q = sygraph::tests::makeQueue();
  auto graph = sygraph::tests::buildGraphFromMatrix(q, sygraph::tests::fixtures::line_5);

  auto lhs = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);
  auto rhs = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);
  auto out = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, graph);

  for (uint v : {0u, 1u, 3u}) { lhs.insert(v); }
  for (uint v : {1u, 2u, 3u}) { rhs.insert(v); }

  auto event = sygraph::operators::intersection::execute(graph, lhs, rhs, out, [](auto) {});
  event.waitAndThrow();

  sygraph::tests::expectFrontier(out, std::vector<uint>{1, 3});

  // Frontiers spanning many bitmap words.
  constexpr size_t n = 1000;
  auto big_graph = sygraph::tests::buildGraph(q, sygraph::tests::gen::randomGraph(n, 2 * n, /*seed=*/29, /*directed=*/false));
  auto make = [&] { return sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_type_t::bitmap>(q, big_graph); };
  auto intersect = [&](auto lhs_pred, auto rhs_pred) {
    auto a = make();
    auto b = make();
    auto result = make();
    sygraph::tests::insertWhere(q, a, lhs_pred);
    sygraph::tests::insertWhere(q, b, rhs_pred);
    sygraph::tests::insertWhere(q, result, [](size_t v) { return v % 11 == 0; }); // stale content must be overwritten
    sygraph::operators::intersection::execute(big_graph, a, b, result, [](auto) {}).waitAndThrow();
    assert(sygraph::tests::readBits(q, result) == sygraph::tests::bitsWhere(n, [=](size_t v) { return lhs_pred(v) && rhs_pred(v); }));
    // The inputs are not modified.
    assert(sygraph::tests::readBits(q, a) == sygraph::tests::bitsWhere(n, lhs_pred));
    assert(sygraph::tests::readBits(q, b) == sygraph::tests::bitsWhere(n, rhs_pred));
  };
  auto multiple_of_3 = [](size_t v) { return v % 3 == 0; };
  intersect(multiple_of_3, [](size_t v) { return v % 2 == 0 || v == n - 1; });         // overlapping
  intersect([](size_t v) { return v % 2 == 0; }, [](size_t v) { return v % 2 == 1; }); // disjoint
  intersect(multiple_of_3, multiple_of_3);                                             // identical
  intersect(multiple_of_3, [](size_t) { return false; });                              // one side empty
}
