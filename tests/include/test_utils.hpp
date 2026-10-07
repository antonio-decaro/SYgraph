#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <queue>
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

// 0 - 1 - 3 and 0 - 2 - 3: two shortest paths from 0 to 3.
inline constexpr std::string_view diamond_4 = "4\n"
                                              "0 1 1 0\n"
                                              "1 0 0 1\n"
                                              "1 0 0 1\n"
                                              "0 1 1 0";

// 0 -10- 1 -10- 2: shortest distances exceed the vertex count.
inline constexpr std::string_view heavy_line_3 = "3\n"
                                                 "0 10 0\n"
                                                 "10 0 10\n"
                                                 "0 10 0";

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

// Deterministic graph generators. They build the CSR directly (sorted adjacency lists, no self-loops, no duplicate
// edges) and use their own PRNG so the graphs are identical with every standard library.
namespace gen {

using csr_t = sygraph::formats::CSR<uint, uint, uint>;

struct Edge {
  uint u;
  uint v;
  uint w;
};

inline uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

// Undirected graphs store every edge in both directions with the same weight.
inline csr_t csrFromEdges(size_t n, std::vector<Edge> edges, bool directed) {
  std::erase_if(edges, [](const Edge& e) { return e.u == e.v; });
  if (!directed) {
    for (auto& e : edges) {
      if (e.u > e.v) { std::swap(e.u, e.v); }
    }
  }
  auto by_endpoints = [](const Edge& a, const Edge& b) { return a.u != b.u ? a.u < b.u : a.v < b.v; };
  auto same_endpoints = [](const Edge& a, const Edge& b) { return a.u == b.u && a.v == b.v; };
  std::stable_sort(edges.begin(), edges.end(), by_endpoints);
  edges.erase(std::unique(edges.begin(), edges.end(), same_endpoints), edges.end());
  if (!directed) {
    const size_t m = edges.size();
    for (size_t i = 0; i < m; ++i) { edges.push_back({edges[i].v, edges[i].u, edges[i].w}); }
    std::sort(edges.begin(), edges.end(), by_endpoints);
  }

  std::vector<uint> offsets(n + 1, 0);
  std::vector<uint> columns;
  std::vector<uint> weights;
  for (const auto& e : edges) {
    offsets[e.u + 1]++;
    columns.push_back(e.v);
    weights.push_back(e.w);
  }
  for (size_t i = 0; i < n; ++i) { offsets[i + 1] += offsets[i]; }
  return csr_t{offsets, columns, weights};
}

inline csr_t randomGraph(size_t n, size_t m, uint64_t seed, bool directed, uint max_weight = 1) {
  std::vector<Edge> edges;
  edges.reserve(m);
  for (size_t i = 0; i < m; ++i) {
    const auto u = static_cast<uint>(splitmix64(seed) % n);
    const auto v = static_cast<uint>(splitmix64(seed) % n);
    const auto w = static_cast<uint>(1 + splitmix64(seed) % max_weight);
    edges.push_back({u, v, w});
  }
  return csrFromEdges(n, std::move(edges), directed);
}

inline csr_t completeGraph(size_t n) {
  std::vector<Edge> edges;
  for (uint u = 0; u < n; ++u) {
    for (uint v = u + 1; v < n; ++v) { edges.push_back({u, v, 1}); }
  }
  return csrFromEdges(n, std::move(edges), /*directed=*/false);
}

// Undirected graph with `components` connected components plus `isolated` vertices without edges. Vertices are
// assigned to components by hash, so a component's largest vertex id is spread over the whole id range.
inline csr_t componentsGraph(size_t n, size_t components, size_t isolated, size_t extra_edges, uint64_t seed) {
  std::vector<std::vector<uint>> members(components);
  for (uint v = 0; v < n; ++v) {
    uint64_t h = v * 0x9e3779b97f4a7c15ULL + seed;
    if (isolated > 0 && v % (n / isolated) == 1) { continue; } // isolated
    members[splitmix64(h) % components].push_back(v);
  }
  std::vector<Edge> edges;
  for (auto& comp : members) {
    // A random spanning path keeps the component connected; extra random edges add cycles.
    for (size_t i = comp.size(); i > 1; --i) { std::swap(comp[i - 1], comp[splitmix64(seed) % i]); }
    for (size_t i = 1; i < comp.size(); ++i) { edges.push_back({comp[i - 1], comp[i], 1}); }
    for (size_t i = 0; i < extra_edges && comp.size() > 1; ++i) {
      edges.push_back({comp[splitmix64(seed) % comp.size()], comp[splitmix64(seed) % comp.size()], 1});
    }
  }
  return csrFromEdges(n, std::move(edges), /*directed=*/false);
}

} // namespace gen

// CPU reference implementations, following the conventions of the library's algorithms.
namespace ref {

using gen::csr_t;

// Hop distances; unreachable vertices get n + 1 (as BFS reports them).
inline std::vector<uint> bfs(const csr_t& csr, uint source) {
  const size_t n = csr.getRowOffsetsSize();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  std::vector<uint> dist(n, static_cast<uint>(n + 1));
  std::vector<uint> queue{source};
  dist[source] = 0;
  for (size_t head = 0; head < queue.size(); ++head) {
    const uint u = queue[head];
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      if (dist[columns[e]] == n + 1) {
        dist[columns[e]] = dist[u] + 1;
        queue.push_back(columns[e]);
      }
    }
  }
  return dist;
}

// Weighted shortest distances; unreachable vertices get UINT_MAX (as SSSP reports them).
inline std::vector<uint> dijkstra(const csr_t& csr, uint source) {
  const size_t n = csr.getRowOffsetsSize();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  const auto& weights = csr.getValues();
  constexpr uint64_t inf = std::numeric_limits<uint64_t>::max();
  std::vector<uint64_t> dist(n, inf);
  std::priority_queue<std::pair<uint64_t, uint>, std::vector<std::pair<uint64_t, uint>>, std::greater<>> pq;
  dist[source] = 0;
  pq.push({0, source});
  while (!pq.empty()) {
    auto [d, u] = pq.top();
    pq.pop();
    if (d != dist[u]) { continue; }
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      if (d + weights[e] < dist[columns[e]]) {
        dist[columns[e]] = d + weights[e];
        pq.push({dist[columns[e]], columns[e]});
      }
    }
  }
  std::vector<uint> result(n);
  for (size_t v = 0; v < n; ++v) {
    assert(dist[v] == inf || dist[v] < std::numeric_limits<uint>::max());
    result[v] = dist[v] == inf ? std::numeric_limits<uint>::max() : static_cast<uint>(dist[v]);
  }
  return result;
}

// Connected-component labels of an undirected graph: the largest vertex id of each component (CC's convention).
inline std::vector<uint> ccMaxLabel(const csr_t& csr) {
  const size_t n = csr.getRowOffsetsSize();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  std::vector<uint> label(n, static_cast<uint>(n));
  for (uint s = 0; s < n; ++s) {
    if (label[s] != n) { continue; }
    std::vector<uint> comp{s};
    label[s] = s;
    for (size_t head = 0; head < comp.size(); ++head) {
      for (size_t e = offsets[comp[head]]; e < offsets[comp[head] + 1]; ++e) {
        if (label[columns[e]] == n) {
          label[columns[e]] = s;
          comp.push_back(columns[e]);
        }
      }
    }
    const uint max_id = *std::max_element(comp.begin(), comp.end());
    for (uint v : comp) { label[v] = max_id; }
  }
  return label;
}

// Single-source Brandes dependency (BC's convention): the fraction of shortest paths from the source to every other
// vertex that pass through each vertex, over unweighted out-edges, without halving. The source gets 0.
inline std::vector<double> brandes(const csr_t& csr, uint source) {
  const size_t n = csr.getRowOffsetsSize();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  const auto dist = bfs(csr, source);
  std::vector<double> sigma(n, 0.0);
  std::vector<uint> order;
  for (uint v = 0; v < n; ++v) {
    if (dist[v] != n + 1) { order.push_back(v); }
  }
  std::sort(order.begin(), order.end(), [&](uint a, uint b) { return dist[a] < dist[b]; });
  sigma[source] = 1.0;
  for (uint u : order) {
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      if (dist[columns[e]] == dist[u] + 1) { sigma[columns[e]] += sigma[u]; }
    }
  }
  std::vector<double> delta(n, 0.0);
  for (auto it = order.rbegin(); it != order.rend(); ++it) {
    const uint u = *it;
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      const uint w = columns[e];
      if (dist[w] == dist[u] + 1) { delta[u] += sigma[u] / sigma[w] * (1.0 + delta[w]); }
    }
  }
  delta[source] = 0.0;
  return delta;
}

// Number of triangles of an undirected graph with sorted adjacency lists.
inline size_t triangles(const csr_t& csr) {
  const size_t n = csr.getRowOffsetsSize();
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  size_t count = 0;
  for (uint u = 0; u < n; ++u) {
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      const uint v = columns[e];
      if (v <= u) { continue; }
      // Common neighbours w > v close the triangle u < v < w.
      size_t i = offsets[u];
      size_t j = offsets[v];
      while (i < offsets[u + 1] && j < offsets[v + 1]) {
        if (columns[i] < columns[j]) {
          ++i;
        } else if (columns[i] > columns[j]) {
          ++j;
        } else {
          count += columns[i] > v;
          ++i;
          ++j;
        }
      }
    }
  }
  return count;
}

} // namespace ref

template<typename T>
void expectNear(const std::vector<T>& actual, const std::vector<double>& expected, double tolerance = 1e-3) {
  assert(actual.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    assert(std::abs(static_cast<double>(actual[i]) - expected[i]) <= tolerance * std::max(1.0, std::abs(expected[i])));
  }
}

template<sygraph::memory::space Space = sygraph::memory::space::shared>
auto buildGraph(sycl::queue& q, const gen::csr_t& csr, sygraph::graph::Properties properties = {}) {
  return sygraph::graph::build::fromCSR<Space>(q, csr, properties);
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
