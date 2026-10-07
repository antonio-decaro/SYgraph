#include "test_utils.hpp"
#include <chrono>
#include <array>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<sygraph::operators::load_balancer LoadBalancer, typename GraphT>
void run_bfs(GraphT& G) {
  using frontier_view_t = sygraph::frontier::frontier_view;
  using frontier_impl_t = sygraph::frontier::frontier_type;

  auto& q = G.getQueue();
  auto in_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);
  auto out_frontier = sygraph::frontier::makeFrontier<frontier_view_t::vertex, frontier_impl_t::bitmap>(q, G);

  bool* visited = sycl::malloc_shared<bool>(G.getVertexCount(), q);
  size_t* distances = sycl::malloc_shared<size_t>(G.getVertexCount(), q);
  for (size_t i = 0; i < G.getVertexCount(); ++i) {
    visited[i] = false;
    distances[i] = 0;
  }

  in_frontier.insert(0);
  distances[0] = 0;
  visited[0] = true;

  while (!in_frontier.empty()) {
    auto e = sygraph::operators::advance::frontier<LoadBalancer, frontier_view_t::vertex, frontier_view_t::vertex>(
        G, in_frontier, out_frontier, [=](auto u, auto v, auto, auto) -> bool {
          if (!visited[v]) {
            visited[v] = true;
            distances[v] = distances[u] + 1;
            return true;
          }
          return false;
        });
    e.waitAndThrow();
    sygraph::frontier::swap(in_frontier, out_frontier);
    out_frontier.clear();
  }

  constexpr std::array<size_t, 6> expected_distances{0, 1, 1, 2, 2, 3};
  for (size_t i = 0; i < G.getVertexCount(); ++i) {
    assert(visited[i]);
    assert(distances[i] == expected_distances[i]);
  }

  sycl::free(visited, q);
  sycl::free(distances, q);
}

int main() {
  auto q = sygraph::tests::makeQueue();

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  using load_balance_t = sygraph::operators::load_balancer;

  auto start = std::chrono::high_resolution_clock::now();

  run_bfs<load_balance_t::workgroup_mapped>(G);
  run_bfs<load_balance_t::bucketing>(G);

  auto end = std::chrono::high_resolution_clock::now();

  std::cout << "Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
}
