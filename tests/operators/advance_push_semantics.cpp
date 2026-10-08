#include "test_utils.hpp"

#include <vector>

#pragma clang diagnostic ignored "-Wdeprecated-declarations" // the bitmap frontier is deprecated but still supported

using sygraph::frontier::frontier_type;
using sygraph::frontier::frontier_view;
using sygraph::operators::load_balancer;
namespace size = sygraph::frontier::size;
using sygraph::tests::bitsWhere;
using sygraph::tests::insertWhere;
using sygraph::tests::readBits;
using sygraph::tests::gen::csr_t;

// Per-edge record of the functor calls made by one advance.
struct Calls {
  sycl::queue& q;
  uint32_t* count;
  uint32_t* src;
  uint32_t* dst;
  size_t edges;

  Calls(sycl::queue& q, size_t edges) : q(q), edges(edges) {
    count = sycl::malloc_shared<uint32_t>(edges, q);
    src = sycl::malloc_shared<uint32_t>(edges, q);
    dst = sycl::malloc_shared<uint32_t>(edges, q);
    q.fill(count, 0U, edges).wait();
  }
  ~Calls() {
    sycl::free(count, q);
    sycl::free(src, q);
    sycl::free(dst, q);
  }
};

// Checks that exactly the out-edges of the active vertices were visited once, with the right endpoints, and that the
// output holds the even destinations of those edges (the functor's choice).
void expectPushResult(const csr_t& csr, const std::vector<bool>& active, const Calls& calls, const std::vector<bool>& output) {
  const auto& offsets = csr.getRowOffsets();
  const auto& columns = csr.getColumnIndices();
  std::vector<bool> expected_output(active.size(), false);
  for (size_t u = 0; u < active.size(); ++u) {
    for (size_t e = offsets[u]; e < offsets[u + 1]; ++e) {
      assert(calls.count[e] == (active[u] ? 1 : 0));
      if (!active[u]) { continue; }
      assert(calls.src[e] == u);
      assert(calls.dst[e] == columns[e]);
      if (columns[e] % 2 == 0) { expected_output[columns[e]] = true; }
    }
  }
  assert(output == expected_output);
}

template<load_balancer Lb, frontier_type FT, typename GraphT, typename PredT>
void runPush(sycl::queue& q, GraphT& graph, const csr_t& csr, PredT pred, size::frontier_size_t expected_size) {
  const size_t n = graph.getVertexCount();
  auto in = sygraph::frontier::makeFrontier<frontier_view::vertex, FT>(q, graph);
  auto out = sygraph::frontier::makeFrontier<frontier_view::vertex, FT>(q, graph);
  insertWhere(q, in, pred);

  Calls calls{q, csr.getNumNonzeros()};
  auto* count = calls.count;
  auto* src_seen = calls.src;
  auto* dst_seen = calls.dst;
  sygraph::operators::advance::frontier<Lb, frontier_view::vertex, frontier_view::vertex>(
      graph,
      in,
      out,
      [=](auto src, auto dst, auto edge, auto) -> bool {
        sygraph::sync::atomicFetchAdd(count + edge, 1U);
        src_seen[edge] = src;
        dst_seen[edge] = dst;
        return dst % 2 == 0;
      },
      expected_size)
      .waitAndThrow();

  expectPushResult(csr, bitsWhere(n, pred), calls, readBits(q, out));
}

// Advance over the whole graph (graph view): every edge is visited exactly once.
template<load_balancer Lb, typename GraphT>
void runVertices(sycl::queue& q, GraphT& graph, const csr_t& csr) {
  auto out = sygraph::frontier::makeFrontier<frontier_view::vertex, frontier_type::mlb>(q, graph);
  Calls calls{q, csr.getNumNonzeros()};
  auto* count = calls.count;
  auto* src_seen = calls.src;
  auto* dst_seen = calls.dst;
  sygraph::operators::advance::vertices<Lb, frontier_view::vertex>(graph, out, [=](auto src, auto dst, auto edge, auto) -> bool {
    sygraph::sync::atomicFetchAdd(count + edge, 1U);
    src_seen[edge] = src;
    dst_seen[edge] = dst;
    return dst % 2 == 0;
  }).waitAndThrow();

  expectPushResult(csr, std::vector<bool>(graph.getVertexCount(), true), calls, readBits(q, out));
}

template<load_balancer Lb, frontier_type FT, typename GraphT>
void runAll(sycl::queue& q, GraphT& graph, const csr_t& csr) {
  const size_t n = graph.getVertexCount();
  const size_t range = sygraph::types::detail::byte_size * sizeof(sygraph::types::bitmap_type_t);
  // A sparse pattern, a run of full words and the last vertex.
  auto pattern = [=](size_t v) { return (v * 2654435761u) % 3 == 0 || v < 2 * range || v == n - 1; };

  for (auto expected_size : {size::fetch_from_memory, size::infer_from_device, 1, static_cast<int>(3 * n)}) {
    runPush<Lb, FT>(q, graph, csr, pattern, expected_size);
  }
  runPush<Lb, FT>(q, graph, csr, [](size_t) { return true; }, size::fetch_from_memory);
  runPush<Lb, FT>(q, graph, csr, [](size_t) { return false; }, size::fetch_from_memory);
}

template<typename GraphT>
void runGraph(sycl::queue& q, GraphT& graph, const csr_t& csr) {
  runAll<load_balancer::workgroup_mapped, frontier_type::mlb>(q, graph, csr);
  runAll<load_balancer::bucketing, frontier_type::mlb>(q, graph, csr);
  runAll<load_balancer::workgroup_mapped, frontier_type::bitmap>(q, graph, csr);
  runAll<load_balancer::bucketing, frontier_type::bitmap>(q, graph, csr);
  runVertices<load_balancer::workgroup_mapped>(q, graph, csr);
  runVertices<load_balancer::bucketing>(q, graph, csr);
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 10000;

  for (bool directed : {false, true}) {
    const auto csr = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/directed ? 7 : 3, directed);
    auto graph = sygraph::tests::buildGraph(q, csr, {.directed = directed});
    runGraph(q, graph, csr);
  }

  // Graph stored in device memory.
  const auto csr = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/11, /*directed=*/false);
  auto device_graph = sygraph::tests::buildGraph<sygraph::memory::space::device>(q, csr);
  runAll<load_balancer::workgroup_mapped, frontier_type::mlb>(q, device_graph, csr);
  runAll<load_balancer::bucketing, frontier_type::mlb>(q, device_graph, csr);

  // A high-degree vertex.
  std::istringstream star{std::string(sygraph::tests::fixtures::star_5)};
  const auto star_csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(star);
  auto star_graph = sygraph::tests::buildGraph(q, star_csr);
  runGraph(q, star_graph, star_csr);
}
