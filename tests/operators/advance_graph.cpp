#include "test_utils.hpp"
#include <chrono>
#include <array>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<sygraph::operators::load_balancer LoadBalancer, typename GraphT>
void run_graph_advance(GraphT& G) {
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_impl_t = sygraph::frontier::frontier_type;

  auto& q = G.getQueue();
  auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::mlb>(q, G);
  bool* visited = sycl::malloc_shared<bool>(G.getVertexCount(), q);
  for (size_t i = 0; i < G.getVertexCount(); ++i) { visited[i] = false; }

  auto device_graph = G.getDeviceGraph();
  sygraph::operators::advance::vertices<LoadBalancer, sygraph::frontier::frontier_view::vertex>(
      G, out_frontier, [=](auto u, auto, auto, auto) -> bool { return device_graph.getDegree(u) != 0; })
      .waitAndThrow();
  for (size_t i = 0; i < G.getVertexCount(); ++i) { visited[i] = out_frontier.check(i); }

  constexpr std::array<bool, 6> expected_visited{true, true, true, true, true, false};
  for (size_t i = 0; i < G.getVertexCount(); ++i) { assert(visited[i] == expected_visited[i]); }

  sycl::free(visited, q);
}

int main() {
  auto q = sygraph::tests::makeQueue();

  auto mat = sygraph::io::storage::matrices::two_cc;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  using load_balance_t = sygraph::operators::load_balancer;

  auto start = std::chrono::high_resolution_clock::now();

  run_graph_advance<load_balance_t::workgroup_mapped>(G);
  run_graph_advance<load_balance_t::bucketing>(G);

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}
