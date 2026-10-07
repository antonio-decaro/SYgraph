#include "test_utils.hpp"

#include <algorithm>
#include <vector>

using frontier_t = sygraph::frontier::Frontier<uint, sygraph::frontier::frontier_type::mlb>;
using sygraph::tests::activeOffsets;
using sygraph::tests::bitsWhere;
using sygraph::tests::expectedWords;
using sygraph::tests::insertWhere;
using sygraph::tests::readBits;

// Checks every observable view of the frontier against the expected element set.
void expectContents(sycl::queue& q, const frontier_t& f, const std::vector<bool>& expected) {
  assert(readBits(q, f) == expected);
  const size_t count = std::count(expected.begin(), expected.end(), true);
  assert(f.size() == count);
  assert(f.empty() == (count == 0));

  const size_t n = expected.size();
  for (size_t i : {size_t{0}, n / 2, n - 1}) { assert(f.check(i) == expected[i]); }
}

void expectActiveFrontier(sycl::queue& q, const frontier_t& f, const std::vector<bool>& expected) {
  f.computeActiveFrontier(false).wait();
  assert(activeOffsets(q, f) == expectedWords(expected, f.getBitmapRange(), false));
  f.computeActiveFrontier(true).wait();
  assert(activeOffsets(q, f) == expectedWords(expected, f.getBitmapRange(), true));
}

void testSize(sycl::queue& q, size_t n) {
  const size_t range = sygraph::types::detail::byte_size * sizeof(sygraph::types::bitmap_type_t);
  // A sparse pattern plus a completely full first word, so both the "has elements" and "is full" cases show up.
  auto pattern = [=](size_t i) { return i % 7 == 0 || i == n - 1 || i < range; };
  auto multiple_of_3 = [](size_t i) { return i % 3 == 0; };
  const auto pattern_bits = bitsWhere(n, pattern);
  const auto empty_bits = std::vector<bool>(n, false);

  frontier_t f{q, n};
  assert(f.getNumElems() == n);
  assert(f.getBitmapSize() == (n + range - 1) / range);
  expectContents(q, f, empty_bits);
  expectActiveFrontier(q, f, empty_bits);

  // Device-side inserts.
  insertWhere(q, f, pattern);
  expectContents(q, f, pattern_bits);
  expectActiveFrontier(q, f, pattern_bits);

  // Removing every element must leave an empty frontier.
  insertWhere(q, f, pattern, /*insert=*/false);
  expectContents(q, f, empty_bits);
  f.computeActiveFrontier(true).wait();
  assert(activeOffsets(q, f) == expectedWords(empty_bits, range, true));

  // Host-side insert/remove of a single element.
  const size_t x = n / 2;
  f.insert(x);
  expectContents(q, f, bitsWhere(n, [=](size_t i) { return i == x; }));
  f.remove(x);
  expectContents(q, f, empty_bits);

  // clear() after inserts.
  insertWhere(q, f, pattern);
  f.clear();
  expectContents(q, f, empty_bits);
  expectActiveFrontier(q, f, empty_bits);

  // saveState / loadState round trip.
  insertWhere(q, f, pattern);
  auto state = f.saveState();
  f.clear();
  f.loadState(state);
  expectContents(q, f, pattern_bits);
  expectActiveFrontier(q, f, pattern_bits);

  // merge into an empty frontier.
  frontier_t merged{q, n};
  merged.merge(f);
  expectContents(q, merged, pattern_bits);
  expectActiveFrontier(q, merged, pattern_bits);

  // merge of two non-empty frontiers.
  frontier_t thirds{q, n};
  insertWhere(q, thirds, multiple_of_3);
  merged.merge(thirds);
  const auto union_bits = bitsWhere(n, [=](size_t i) { return pattern(i) || multiple_of_3(i); });
  expectContents(q, merged, union_bits);
  expectActiveFrontier(q, merged, union_bits);

  // intersect: the overlap of the pattern and the multiples of 3.
  frontier_t intersected{q, n};
  intersected.merge(f);
  intersected.intersect(thirds);
  const auto intersection_bits = bitsWhere(n, [=](size_t i) { return pattern(i) && multiple_of_3(i); });
  expectContents(q, intersected, intersection_bits);
  // Words that became empty may still be reported by the non-inverted active frontier, but every word with an element
  // must be there.
  intersected.computeActiveFrontier(false).wait();
  auto offsets = activeOffsets(q, intersected);
  for (int w : expectedWords(intersection_bits, range)) { assert(std::binary_search(offsets.begin(), offsets.end(), w)); }

  // intersect with a disjoint frontier leaves it empty.
  frontier_t odd{q, n};
  insertWhere(q, odd, [](size_t i) { return i % 2 == 1; });
  frontier_t even{q, n};
  insertWhere(q, even, [](size_t i) { return i % 2 == 0; });
  even.intersect(odd);
  expectContents(q, even, empty_bits);

  // swap exchanges the contents.
  frontier_t a{q, n};
  frontier_t b{q, n};
  insertWhere(q, a, pattern);
  insertWhere(q, b, multiple_of_3);
  sygraph::frontier::swap(a, b);
  expectContents(q, a, bitsWhere(n, multiple_of_3));
  expectContents(q, b, pattern_bits);
  expectActiveFrontier(q, a, bitsWhere(n, multiple_of_3));
}

int main() {
  auto q = sygraph::tests::makeQueue();
  for (size_t n : {1, 31, 32, 33, 64, 65, 1000, 4097, 100000}) { testSize(q, n); }
}
