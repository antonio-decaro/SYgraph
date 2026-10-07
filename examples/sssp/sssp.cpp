/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../include/utils.hpp"
#include <CLI/CLI.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>
#include <utility>
#include <vector>

template<typename VertexT, typename WeightT>
class Prioritize {
public:
  bool operator()(std::pair<VertexT, WeightT>& p1, std::pair<VertexT, WeightT>& p2) { return p1.second > p2.second; }
};

template<typename GraphT, typename BenchT>
bool validate(const GraphT& graph, BenchT& sssp, uint source) {
  using vertex_t = typename GraphT::vertex_t;
  using weight_t = typename GraphT::weight_t;
  using edge_t = typename GraphT::edge_t;
  auto* row_offsets = graph.getRowOffsets();
  auto* column_indices = graph.getColumnIndices();
  auto* nonzero_values = graph.getValues();

  std::vector<weight_t> distances(graph.getVertexCount(), BenchT::unreachable());
  distances[source] = 0;

  std::priority_queue<std::pair<vertex_t, weight_t>, std::vector<std::pair<vertex_t, weight_t>>, Prioritize<vertex_t, weight_t>> pq;
  pq.push(std::make_pair(source, 0.0));

  while (!pq.empty()) {
    std::pair<vertex_t, weight_t> curr = pq.top();
    pq.pop();

    vertex_t curr_node = curr.first;
    weight_t curr_dist = curr.second;

    vertex_t start = row_offsets[curr_node];
    vertex_t end = row_offsets[curr_node + 1];

    for (vertex_t offset = start; offset < end; offset++) {
      vertex_t neib = column_indices[offset];
      weight_t new_dist = curr_dist + nonzero_values[offset];
      if (new_dist < distances[neib]) {
        distances[neib] = new_dist;
        pq.push(std::make_pair(neib, new_dist));
      }
    }
  }

  const auto computed = sssp.getDistances();
  for (auto i = 0; i < graph.getVertexCount(); i++) {
    if (distances[i] != computed[i]) {
      std::cerr << "Mismatch at vertex " << i << " | Expected: " << distances[i] << " | Got: " << computed[i] << std::endl;
      return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  using type_t = unsigned int;
  GraphOptions opts;
  CLI::App app{"SYgraph example"};
  auto source_option = configureBaseCLI(app, opts);
  CLI11_PARSE(app, argc, argv);
  finalizeGraphOptions(opts, source_option);

  std::cerr << "[*] Reading CSR" << std::endl;
  sygraph::graph::Properties properties;
  auto csr = readCSR<float, type_t, type_t>(opts, &properties);

#ifdef ENABLE_PROFILING
  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::enable_profiling()};
#else
  sycl::queue q{sycl::gpu_selector_v};
#endif

  std::cerr << "[*] Building Graph" << std::endl;
  auto G = sygraph::graph::build::fromCSR<graph_location>(q, csr, properties);
  printGraphInfo(G);
  size_t size = G.getVertexCount();

  sygraph::algorithms::SSSP sssp{G};
  if (opts.random_source) { opts.source = getRandomSource(size); }
  type_t sssp_source = static_cast<type_t>(opts.source);
  sssp.init(sssp_source);

  std::cout << "[*] Running SSSP on source " << opts.source << std::endl;
  sssp.run<true>();

  std::cerr << "[!] Done" << std::endl;

  if (opts.validate) {
    std::cout << "Validation: [";
    auto validation_start = std::chrono::high_resolution_clock::now();
    if (!validate(G, sssp, opts.source)) {
      std::cout << failString();
    } else {
      std::cout << successString();
    }
    std::cout << "] | ";
    auto validation_end = std::chrono::high_resolution_clock::now();
    std::cout << "Validation Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(validation_end - validation_start).count() << " ms"
              << std::endl;
  }

  if (opts.print_output) {
    std::cout << std::left;
    std::cout << std::setw(10) << "Vertex" << std::setw(10) << "Distance" << std::endl;
    for (size_t i = 0; i < G.getVertexCount(); i++) {
      auto distance = sssp.getDistance(i);
      if (distance != size + 1) { std::cout << std::setw(10) << i << std::setw(10) << distance << std::endl; }
    }
  }

  printProfilingOutput(opts);
  // Profiling events must be released before queue/runtime teardown at exit.
  clearProfilingOutput();
}
