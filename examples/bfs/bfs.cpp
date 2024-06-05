#include "../include/utils.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

template<typename GraphT, typename BfsT>
bool validate(const GraphT& graph, BfsT& bfs, uint source) {
  using vertex_t = typename GraphT::vertex_t;
  assert(bfs.getDistance(source) == 0);
  std::vector<uint> distances(graph.getVertexCount(), graph.getVertexCount() + 1);
  std::vector<vertex_t> in_frontier;
  std::vector<vertex_t> out_frontier;
  in_frontier.push_back(source);
  distances[source] = 0;

  auto row_offsets = graph.getRowOffsets();
  auto col_indices = graph.getColumnIndices();

  size_t iter = 0;
  size_t mismatches = 0;
  while (in_frontier.size()) {
    for (size_t i = 0; i < in_frontier.size(); i++) {
      auto vertex = in_frontier[i];

      auto start = row_offsets[vertex];
      auto end = row_offsets[vertex + 1];

      for (size_t j = start; j < end; j++) {
        auto neighbor = col_indices[j];
        if (distances[neighbor] == graph.getVertexCount() + 1) {
          distances[neighbor] = distances[vertex] + 1;
          if (distances[neighbor] != bfs.getDistance(neighbor)) {
            // std::cout << "Distance mismatch at vertex " << neighbor << " expected " << distances[neighbor] << " got " << bfs.getDistance(neighbor)
            // << std::endl;
            mismatches++;
          }
          out_frontier.push_back(neighbor);
        }
      }
    }
    std::swap(in_frontier, out_frontier);
    out_frontier.clear();
    iter++;
  }
  if (mismatches) { std::cout << "Mismatches: " << mismatches << std::endl; }
  return mismatches == 0;
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  args_t<type_t> args{argc, argv};

  std::cerr << "[*] Reading CSR" << std::endl;
  auto csr = read_csr<type_t, type_t, type_t>(args);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  print_device_info(q, "[*] ");

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);
  print_graph_info(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::BFS bfs{G};
  if (args.random_source) { args.source = get_random_source(size); }
  bfs.init(args.source);

  std::cerr << "[*] Running BFS on source " << args.source << std::endl;
  bfs.run<true>();

  std::cerr << "[!] Done" << std::endl;

  if (args.validate) {
    std::cerr << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, bfs, args.source)) {
      std::cerr << "\033[1;31mFailed\033[0m";
    } else {
      std::cerr << "\033[1;32mSuccess\033[0m";
    }
    std::cerr << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cerr << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (args.print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      auto distance = bfs.getDistance(i);
      if (distance != size + 1) { std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl; }
    }
  }

#ifdef ENABLE_PROFILING
  sygraph::profiler::print();
#endif
}
