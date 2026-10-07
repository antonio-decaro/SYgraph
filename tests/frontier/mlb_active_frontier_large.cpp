#include "test_utils.hpp"

#include <numeric>
#include <vector>

using frontier_t = sygraph::frontier::Frontier<uint, sygraph::frontier::frontier_type::mlb>;
using sygraph::tests::activeOffsets;
using sygraph::tests::insertWhere;

// computeActiveFrontier launches a fixed number of work-groups (one per compute unit). This frontier is large enough
// that every work-item has to scan at least two level-1 words, and every work-group finds active words.
int main() {
  auto q = sygraph::tests::makeQueue();

  const size_t range = sygraph::types::detail::byte_size * sizeof(sygraph::types::bitmap_type_t);
  const size_t num_cus = sygraph::detail::device::getNumComputeUnits(q);
  const size_t global_size = num_cus * sygraph::types::detail::COMPUTE_UNIT_SIZE;
  const size_t n = 2 * global_size * range * range + range + 1; // the last level-0 word is partially used
  const size_t words = (n + range - 1) / range;

  frontier_t f{q, n};

  // Every element active: every word is reported, and only the partial last word is "not full".
  insertWhere(q, f, [](size_t) { return true; });
  std::vector<int> all_words(words);
  std::iota(all_words.begin(), all_words.end(), 0);
  for (int repeat = 0; repeat < 2; ++repeat) { // the second call must not accumulate on the first one's result
    f.computeActiveFrontier(false).wait();
    assert(activeOffsets(q, f) == all_words);
  }
  f.computeActiveFrontier(true).wait();
  assert(activeOffsets(q, f) == std::vector<int>{static_cast<int>(words - 1)});

  // Every fifth word active.
  f.clear();
  insertWhere(q, f, [=](size_t i) { return (i / range) % 5 == 0; });
  std::vector<int> every_fifth;
  for (size_t w = 0; w < words; w += 5) { every_fifth.push_back(static_cast<int>(w)); }
  for (int repeat = 0; repeat < 2; ++repeat) {
    f.computeActiveFrontier(false).wait();
    assert(activeOffsets(q, f) == every_fifth);
  }
  size_t expected_size = 0;
  for (size_t i = 0; i < n; ++i) { expected_size += (i / range) % 5 == 0; }
  assert(f.size() == expected_size);
}
