#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include <sycl/sycl.hpp>
#include <sygraph/io/matrices.hpp>
#include <sygraph/sygraph.hpp>

namespace sygraph::tests {

namespace fixtures {

inline constexpr std::string_view line_5 = "5\n"
                                           "0 1 0 0 0\n"
                                           "1 0 1 0 0\n"
                                           "0 1 0 1 0\n"
                                           "0 0 1 0 1\n"
                                           "0 0 0 1 0";

inline constexpr std::string_view star_5 = "5\n"
                                           "0 1 1 1 1\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0\n"
                                           "1 0 0 0 0";

inline constexpr std::string_view triangle_3 = "3\n"
                                               "0 1 1\n"
                                               "1 0 1\n"
                                               "1 1 0";

inline constexpr std::string_view complete_4 = "4\n"
                                               "0 1 1 1\n"
                                               "1 0 1 1\n"
                                               "1 1 0 1\n"
                                               "1 1 1 0";

inline constexpr std::string_view weighted_directed_5 = "5\n"
                                                        "0 1 4 0 0\n"
                                                        "0 0 2 6 0\n"
                                                        "0 0 0 1 5\n"
                                                        "0 0 0 0 1\n"
                                                        "0 0 0 0 0";

} // namespace fixtures

// Prefers a GPU and falls back to any available device (e.g., a CPU). Use ONEAPI_DEVICE_SELECTOR to pin a backend.
// When no device is available the test is skipped, unless SYGRAPH_TEST_REQUIRE_DEVICE is set (as in CI), in which
// case it fails so that a misconfigured environment cannot report green.
inline sycl::queue makeQueue() {
  try {
    return sycl::queue{sycl::gpu_selector_v};
  } catch (const sycl::exception&) {
    try {
      return sycl::queue{sycl::default_selector_v};
    } catch (const sycl::exception&) {
      if (std::getenv("SYGRAPH_TEST_REQUIRE_DEVICE") != nullptr) {
        std::cerr << "No SYCL device available and SYGRAPH_TEST_REQUIRE_DEVICE is set" << std::endl;
        std::exit(1);
      }
      std::cout << "Skipping test: no SYCL platform available" << std::endl;
      std::exit(0);
    }
  }
}

template<sygraph::memory::space Space = sygraph::memory::space::shared, typename ValueT = uint, typename IndexT = uint, typename OffsetT = uint>
auto buildGraphFromMatrix(sycl::queue& q, std::string_view matrix, sygraph::graph::Properties properties = {}) {
  std::istringstream iss{std::string(matrix)};
  auto csr = sygraph::io::csr::fromMatrix<ValueT, IndexT, OffsetT>(iss);
  return sygraph::graph::build::fromCSR<Space>(q, std::move(csr), properties);
}

template<typename T, size_t N>
void expectEqual(const std::vector<T>& actual, const std::array<T, N>& expected) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) { assert(actual[i] == expected[i]); }
}

template<typename T>
void expectEqual(const std::vector<T>& actual, const std::vector<T>& expected) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) { assert(actual[i] == expected[i]); }
}

template<typename FrontierT>
std::vector<typename FrontierT::type_t> activeElements(const FrontierT& frontier) {
  using value_t = typename FrontierT::type_t;

  std::vector<value_t> values;
  for (size_t i = 0; i < frontier.getNumElems(); ++i) {
    if (frontier.check(i)) { values.push_back(static_cast<value_t>(i)); }
  }
  return values;
}

template<typename FrontierT>
void expectFrontier(const FrontierT& frontier, const std::vector<typename FrontierT::type_t>& expected) {
  expectEqual(activeElements(frontier), expected);
}

// Reads the element bits of a frontier (MLB level 0 or the bitmap) with a single copy. Unlike activeElements(), which
// calls check() and launches a kernel per element, this is cheap enough for frontiers with many elements.
template<typename FrontierT>
std::vector<bool> readBits(sycl::queue& q, const FrontierT& frontier) {
  using bitmap_t = typename FrontierT::bitmap_type;
  const size_t range = frontier.getBitmapRange();
  std::vector<bitmap_t> words(frontier.getBitmapSize());
  q.copy(frontier.getDeviceFrontier().getData(), words.data(), words.size()).wait();

  std::vector<bool> bits(frontier.getNumElems());
  for (size_t i = 0; i < bits.size(); ++i) { bits[i] = (words[i / range] >> (i % range)) & 1; }
  return bits;
}

// Inserts (or removes) on the device every element i for which pred(i) is true.
template<typename FrontierT, typename PredT>
void insertWhere(sycl::queue& q, const FrontierT& frontier, PredT pred, bool insert = true) {
  auto bitmap = frontier.getDeviceFrontier();
  q.parallel_for(sycl::range<1>{frontier.getNumElems()}, [=](sycl::id<1> idx) {
     if (!pred(idx[0])) { return; }
     if (insert) {
       bitmap.insert(idx[0]);
     } else {
       bitmap.remove(idx[0]);
     }
   }).wait();
}

template<typename PredT>
std::vector<bool> bitsWhere(size_t n, PredT pred) {
  std::vector<bool> bits(n);
  for (size_t i = 0; i < n; ++i) { bits[i] = pred(i); }
  return bits;
}

// Returns the word offsets produced by the last computeActiveFrontier() call, sorted.
template<typename FrontierT>
std::vector<int> activeOffsets(sycl::queue& q, const FrontierT& frontier) {
  auto bitmap = frontier.getDeviceFrontier();
  uint32_t count = 0;
  q.copy(bitmap.getOffsetsSize(), &count, 1).wait();
  std::vector<int> offsets(count);
  if (count > 0) { q.copy(bitmap.getOffsets(), offsets.data(), count).wait(); }
  std::sort(offsets.begin(), offsets.end());
  return offsets;
}

// Word offsets an active-frontier computation must report: words with at least one element, or, for pull (invert),
// words that are not completely full.
inline std::vector<int> expectedWords(const std::vector<bool>& bits, size_t range, bool invert = false) {
  std::vector<int> words;
  for (size_t w = 0; w * range < bits.size(); ++w) {
    size_t active = 0;
    for (size_t i = w * range; i < std::min(bits.size(), (w + 1) * range); ++i) { active += bits[i]; }
    if (invert ? active < range : active > 0) { words.push_back(static_cast<int>(w)); }
  }
  return words;
}

} // namespace sygraph::tests
