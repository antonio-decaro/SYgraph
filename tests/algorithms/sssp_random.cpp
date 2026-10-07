#include "test_utils.hpp"

#include <vector>

template<typename GraphT>
void runAll(sycl::queue& q, GraphT& graph, const sygraph::tests::gen::csr_t& csr, std::initializer_list<uint> sources) {
  for (uint source : sources) {
    sygraph::algorithms::SSSP sssp(graph);
    sssp.init(source);
    sssp.run();
    assert(sssp.getDistances() == sygraph::tests::ref::dijkstra(csr, source));
  }
}

int main() {
  auto q = sygraph::tests::makeQueue();
  constexpr size_t n = 5000;
  const sygraph::graph::Properties undirected_props{.directed = false, .weighted = true};
  const sygraph::graph::Properties directed_props{.directed = true, .weighted = true};

  const auto undirected = sygraph::tests::gen::randomGraph(n, 4 * n, /*seed=*/43, /*directed=*/false, /*max_weight=*/1000);
  auto undirected_graph = sygraph::tests::buildGraph(q, undirected, undirected_props);
  runAll(q, undirected_graph, undirected, {0, n / 2});

  // Sparse enough that some vertices are unreachable.
  const auto directed = sygraph::tests::gen::randomGraph(n, 2 * n, /*seed=*/47, /*directed=*/true, /*max_weight=*/1000);
  auto directed_graph = sygraph::tests::buildGraph(q, directed, directed_props);
  runAll(q, directed_graph, directed, {0, n / 2});

  auto device_graph = sygraph::tests::buildGraph<sygraph::memory::space::device>(q, directed, directed_props);
  runAll(q, device_graph, directed, {1});
}
