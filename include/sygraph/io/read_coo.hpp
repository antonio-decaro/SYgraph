/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace io {
namespace coo {

/**
 * Retrieves the COO (Coordinate List) representation of a graph.
 *
 * The input starts with optional `%` comment lines and a header `n n m` (vertex count, repeated, and number of
 * edges), followed by one edge per line: `u v [w]`, with 0-based vertex ids and an optional weight (1 when omitted).
 * Blank lines and `%` comment lines are skipped.
 *
 * @tparam ValueT The value type of the graph.
 * @tparam IndexT The index type of the graph.
 * @tparam OffsetT The offset type of the graph.
 *
 * @param iss The input stream containing the COO representation of the graph.
 * @param undirected Whether the graph is undirected. If true, every edge is also stored in the opposite direction.
 * @param properties If not null, receives whether the graph is directed and whether it has explicit weights.
 * @return The COO representation of the graph, whose vertex count is taken from the header.
 *
 * @throws std::runtime_error If the header is missing or malformed, the vertex counts differ, an edge line is malformed
 * or refers to a vertex out of range, or the number of edge lines differs from the header.
 */
template<typename ValueT, typename IndexT, typename OffsetT = types::offset_t>
sygraph::formats::COO<ValueT, IndexT, OffsetT> fromCOO(std::istream& iss, bool undirected = false, sygraph::graph::Properties* properties = nullptr) {
  auto skippable = [](const std::string& line) { return line.find_first_not_of(" \t\r") == std::string::npos || line[0] == '%'; };

  std::string line;
  do {
    if (!std::getline(iss, line)) { throw std::runtime_error("Error: could not read the first line of the file."); }
  } while (skippable(line));

  size_t n_nodes1 = 0;
  size_t n_nodes2 = 0;
  size_t num_edges = 0;
  if (!(std::istringstream{line} >> n_nodes1 >> n_nodes2 >> num_edges)) {
    throw std::runtime_error("Malformed header in COO file: \"" + line + "\"");
  }
  if (n_nodes1 != n_nodes2) {
    throw std::runtime_error("The COO graph must be square, got " + std::to_string(n_nodes1) + "x" + std::to_string(n_nodes2));
  }

  std::vector<IndexT> coo_row_indices;
  std::vector<IndexT> coo_col_indices;
  std::vector<ValueT> coo_values;
  const size_t stored_edges = num_edges * (undirected ? 2 : 1);
  coo_row_indices.reserve(stored_edges);
  coo_col_indices.reserve(stored_edges);
  coo_values.reserve(stored_edges);

  bool weighted = false;
  size_t edge_lines = 0;
  while (std::getline(iss, line)) {
    if (skippable(line)) { continue; }
    std::istringstream liss{line};
    IndexT u;
    IndexT v;
    ValueT w = static_cast<ValueT>(1);
    if (!(liss >> u >> v)) { throw std::runtime_error("Malformed edge in COO file: \"" + line + "\""); }
    if (liss >> w) {
      weighted = true; // explicit weight column provided
    } else if (!liss.eof()) {
      throw std::runtime_error("Malformed edge in COO file: \"" + line + "\"");
    }
    if (u >= n_nodes1 || v >= n_nodes1) {
      throw std::runtime_error("Vertex out of range in COO file (" + std::to_string(n_nodes1) + " vertices): \"" + line + "\"");
    }
    ++edge_lines;
    coo_row_indices.push_back(u);
    coo_col_indices.push_back(v);
    coo_values.push_back(w);
    if (undirected) {
      coo_row_indices.push_back(v);
      coo_col_indices.push_back(u);
      coo_values.push_back(w);
    }
  }
  if (edge_lines != num_edges) {
    throw std::runtime_error("The COO file has " + std::to_string(edge_lines) + " edges, expected " + std::to_string(num_edges));
  }

  if (properties) {
    properties->directed = !undirected;
    properties->weighted = weighted;
  }

  return sygraph::formats::COO<ValueT, IndexT, OffsetT>(coo_row_indices, coo_col_indices, coo_values, n_nodes1);
}

} // namespace coo
} // namespace io
} // namespace sygraph
