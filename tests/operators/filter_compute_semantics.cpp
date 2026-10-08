#include "test_utils.hpp"

#include <limits>
#include <vector>

using sygraph::frontier::frontier_type;
using sygraph::frontier::frontier_view;
using sygraph::tests::bitsWhere;
using sygraph::tests::insertWhere;
using sygraph::tests::readBits;
namespace compute = sygraph::operators::compute;
namespace filter = sygraph::operators::filter;

// Runs compute::execute and returns how many times each vertex was visited.
template<typename GraphT, typename FrontierT>
std::vector<uint32_t> executeCounts(sycl::queue& q, GraphT& graph, const FrontierT& frontier) {
  const size_t n = graph.getVertexCount();
  uint32_t* counts = sycl::malloc_shared<uint32_t>(n, q);
  q.fill(counts, 0U, n).wait();
  compute::execute<frontier_view::vertex>(graph, frontier, [=](auto v) { sygraph::sync::atomicFetchAdd(counts + v, 1U); }).waitAndThrow();
  std::vector<uint32_t> result(counts, counts + n);
  sycl::free(counts, q);
  return result;
}

void expectVisitedOnce(const std::vector<uint32_t>& counts, const std::vector<bool>& expected) {
  for (size_t v = 0; v < counts.size(); ++v) { assert(counts[v] == (expected[v] ? 1U : 0U)); }
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 10000;
  const auto csr = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/23, /*directed=*/false);
  auto graph = sygraph::tests::buildGraph(q, csr);
  auto make = [&] { return sygraph::frontier::makeFrontier<frontier_view::vertex, frontier_type::mlb>(q, graph); };

  auto pattern = [](size_t v) { return (v * 2654435761u) % 3 == 0 || v == n - 1; };
  auto even = [](size_t v) { return v % 2 == 0; };
  const auto pattern_bits = bitsWhere(n, pattern);

  // external keeps the elements for which the predicate is true, and leaves the input untouched.
  auto in = make();
  auto out = make();
  insertWhere(q, in, pattern);
  insertWhere(q, out, [](size_t v) { return v % 5 == 0; }); // stale content must be cleared
  filter::external(graph, in, out, [](auto v) { return v % 2 == 0; }).waitAndThrow();
  assert(readBits(q, out) == bitsWhere(n, [=](size_t v) { return pattern(v) && even(v); }));
  assert(readBits(q, in) == pattern_bits);

  // inplace removes the elements for which the predicate is true.
  filter::inplace(graph, in, [](auto v) { return v % 2 == 0; }).waitAndThrow();
  assert(readBits(q, in) == bitsWhere(n, [=](size_t v) { return pattern(v) && !even(v); }));

  // inplace removing everything leaves an empty frontier that compute does not visit.
  filter::inplace(graph, in, [](auto) { return true; }).waitAndThrow();
  assert(in.empty());
  assert(in.size() == 0);
  expectVisitedOnce(executeCounts(q, graph, in), std::vector<bool>(n, false));

  // execute visits every active vertex exactly once.
  auto f = make();
  insertWhere(q, f, pattern);
  expectVisitedOnce(executeCounts(q, graph, f), pattern_bits);

  // execute on a frontier filled by merge.
  auto merged = make();
  merged.merge(f);
  expectVisitedOnce(executeCounts(q, graph, merged), pattern_bits);

  // reduce honours the reduction operator; the accumulator's initial value takes part in the reduction.
  uint64_t expected_sum = 7;
  uint32_t expected_max = 0;
  uint32_t expected_min = std::numeric_limits<uint32_t>::max();
  for (size_t v = 0; v < n; ++v) {
    if (!pattern(v)) { continue; }
    expected_sum += v;
    expected_max = std::max<uint32_t>(expected_max, v);
    expected_min = std::min<uint32_t>(expected_min, v);
  }
  uint64_t sum = 7;
  compute::reduce<frontier_view::vertex, sycl::plus<uint64_t>>(graph, f, sum, [](auto v, auto& acc) { acc += v; }).waitAndThrow();
  assert(sum == expected_sum);

  uint32_t max = 0;
  compute::reduce<frontier_view::vertex, sycl::maximum<uint32_t>>(graph, f, max, [](auto v, auto& acc) { acc.combine(v); }).waitAndThrow();
  assert(max == expected_max);

  uint32_t min = std::numeric_limits<uint32_t>::max();
  compute::reduce<frontier_view::vertex, sycl::minimum<uint32_t>>(graph, f, min, [](auto v, auto& acc) { acc.combine(v); }).waitAndThrow();
  assert(min == expected_min);

  auto small = make();
  for (uint v : {1u, 2u, 3u, 4u, 5u}) { small.insert(v); }
  uint64_t product = 1;
  compute::reduce<frontier_view::vertex, sycl::multiplies<uint64_t>>(graph, small, product, [](auto v, auto& acc) {
    acc.combine(static_cast<uint64_t>(v));
  }).waitAndThrow();
  assert(product == 120);

  // Empty frontier: no visits, accumulator unchanged, empty filter output.
  auto empty = make();
  expectVisitedOnce(executeCounts(q, graph, empty), std::vector<bool>(n, false));
  uint64_t untouched = 42;
  compute::reduce<frontier_view::vertex, sycl::plus<uint64_t>>(graph, empty, untouched, [](auto v, auto& acc) { acc += v; }).waitAndThrow();
  assert(untouched == 42);
  filter::external(graph, empty, out, [](auto) { return true; }).waitAndThrow();
  assert(out.empty());
}
