#include "test_utils.hpp"

#include <vector>

// FrontierBitmap is deprecated in favour of MLB but still supported.
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

using frontier_t = sygraph::frontier::Frontier<uint, sygraph::frontier::frontier_type::bitmap>;
using sygraph::tests::activeOffsets;
using sygraph::tests::bitsWhere;
using sygraph::tests::expectedWords;
using sygraph::tests::insertWhere;
using sygraph::tests::readBits;

void expectContents(sycl::queue& q, const frontier_t& f, const std::vector<bool>& expected) {
  assert(readBits(q, f) == expected);
  const size_t count = std::count(expected.begin(), expected.end(), true);
  assert(f.getNumActiveElements() == count);
  assert(f.empty() == (count == 0));
  for (size_t i = 0; i < expected.size(); ++i) { assert(f.check(i) == expected[i]); }
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 1000;
  const size_t range = sygraph::types::detail::byte_size * sizeof(sygraph::types::bitmap_type_t);
  auto pattern = [](size_t i) { return i % 7 == 0 || i == n - 1; };
  auto multiple_of_3 = [](size_t i) { return i % 3 == 0; };
  const auto empty_bits = std::vector<bool>(n, false);

  frontier_t f{q, n};
  expectContents(q, f, empty_bits);

  insertWhere(q, f, pattern);
  expectContents(q, f, bitsWhere(n, pattern));
  assert(f.computeActiveFrontier() == expectedWords(bitsWhere(n, pattern), range).size());
  assert(activeOffsets(q, f) == expectedWords(bitsWhere(n, pattern), range));

  f.remove(0);
  f.remove(n - 1);
  expectContents(q, f, bitsWhere(n, [=](size_t i) { return pattern(i) && i != 0 && i != n - 1; }));

  // Only the top bit of two words is set: empty() must not be fooled by the words summing to zero.
  f.clear();
  f.insert(range - 1);
  f.insert(2 * range - 1);
  expectContents(q, f, bitsWhere(n, [=](size_t i) { return i == range - 1 || i == 2 * range - 1; }));

  // The active frontier must reflect inserts made after an earlier computation, without a clear() in between.
  f.clear();
  f.insert(3);
  assert(f.computeActiveFrontier() == 1);
  f.insert(n - 1);
  assert(f.computeActiveFrontier() == 2);
  assert(activeOffsets(q, f) == (std::vector<int>{0, static_cast<int>((n - 1) / range)}));

  // merge and intersect.
  frontier_t a{q, n};
  frontier_t b{q, n};
  insertWhere(q, a, pattern);
  insertWhere(q, b, multiple_of_3);
  a.merge(b).waitAndThrow();
  expectContents(q, a, bitsWhere(n, [=](size_t i) { return pattern(i) || multiple_of_3(i); }));
  a.intersect(b).waitAndThrow();
  expectContents(q, a, bitsWhere(n, multiple_of_3));

  // swap.
  frontier_t c{q, n};
  insertWhere(q, c, pattern);
  sygraph::frontier::swap(a, c);
  expectContents(q, a, bitsWhere(n, pattern));
  expectContents(q, c, bitsWhere(n, multiple_of_3));

  c.clear();
  expectContents(q, c, empty_bits);
}
