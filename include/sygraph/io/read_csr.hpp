/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <sycl/sycl.hpp>

#include <sygraph/formats/coo.hpp>
#include <sygraph/formats/csr.hpp>
#include <sygraph/graph/properties.hpp>
#include <sygraph/io/matrix_market.hpp>

namespace sygraph {
namespace io {
namespace csr {
namespace detail {
namespace binary {
static constexpr uint64_t magic = 0x5359475243535201ULL; // "SYGRCSR" + version marker
static constexpr uint8_t directed_mask = 0x1;
static constexpr uint8_t weighted_mask = 0x2;
} // namespace binary
} // namespace detail

/**
 * @brief Converts a matrix in CSR format to a CSR object.
 *
 * This function reads a matrix in CSR format from a given input stream and converts it into a CSR object.
 * The CSR object contains the row offsets, column indices, and non-zero values of the matrix.
 *
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param iss The input stream containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMatrix(std::istream& iss) {
  size_t n_rows = 0;
  size_t n_nonzeros = 0;
  std::vector<OffsetT> row_offsets;
  std::vector<IndexT> column_indices;
  std::vector<ValueT> nnz_values;

  // Read number of rows
  iss >> n_rows;

  row_offsets.push_back(0);

  // Read adjacency matrix
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_rows; ++j) {
      ValueT value;
      iss >> value;
      if (value != static_cast<ValueT>(0)) {
        nnz_values.push_back(value);
        column_indices.push_back(j);
      }
    }
    row_offsets.push_back(nnz_values.size());
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Reads a Matrix Market file in coordinate format and converts it to a CSR matrix.
 *
 * This function parses a Matrix Market file from an input stream and converts it to a CSR (Compressed Sparse Row)
 * matrix format. The function only supports `coordinate` format and `general` or `symmetric` symmetry.
 * If the matrix is symmetric, both (row, col) and (col, row) entries are stored for each non-diagonal entry.
 * The entries of each row are sorted by column. Comment lines and blank lines are skipped.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is int).
 * @tparam OffsetT Type of the offsets (default is int).
 *
 * @param iss Input stream containing the Matrix Market data.
 * @return CSR<ValueT, IndexT, OffsetT> An instance of the CSR class containing the matrix data in CSR format.
 *
 * @throws std::runtime_error if the banner is missing or invalid, the format, field or symmetry is unsupported, the
 * matrix is not square, an entry is malformed or out of range, or the number of entries differs from the size line.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMM(std::istream& iss, sygraph::graph::Properties* properties = nullptr) {
  sygraph::io::detail::mm::Banner banner;

  size_t rows = 0, cols = 0, nnz = 0;
  size_t read_entries = 0;
  std::vector<std::tuple<IndexT, IndexT, ValueT>> entries;

  std::string line;
  bool dimensions_read = false;

  if (!std::getline(iss, line) || line.rfind("%%", 0) != 0) { throw std::runtime_error("Missing MatrixMarket banner"); }
  banner.read(line);
  banner.validate<ValueT, IndexT, OffsetT>();
  if (properties) {
    properties->directed = !banner.isSymmetric();
    properties->weighted = !banner.isPattern();
  }

  while (std::getline(iss, line)) {
    // Skip comments and blank lines.
    if (line.empty() || line[0] == '%' || line.find_first_not_of(" \t\r") == std::string::npos) { continue; }
    std::istringstream line_stream(line);

    if (!dimensions_read) {
      if (!(line_stream >> rows >> cols >> nnz)) { throw std::runtime_error("Malformed MatrixMarket size line: \"" + line + "\""); }
      if (rows != cols) {
        throw std::runtime_error("The MatrixMarket matrix must be square, got " + std::to_string(rows) + "x" + std::to_string(cols));
      }
      dimensions_read = true;
      continue;
    }

    size_t row = 0;
    size_t col = 0;
    ValueT value = static_cast<ValueT>(1);
    if (!(line_stream >> row >> col) || (!banner.isPattern() && !(line_stream >> value))) {
      throw std::runtime_error("Malformed MatrixMarket entry: \"" + line + "\"");
    }
    if (row < 1 || row > rows || col < 1 || col > cols) {
      throw std::runtime_error("MatrixMarket entry out of range (" + std::to_string(rows) + " rows): \"" + line + "\"");
    }
    ++read_entries;

    entries.emplace_back(static_cast<IndexT>(row - 1), static_cast<IndexT>(col - 1), value);
    // For symmetric matrices, also add the transpose entry if not on the diagonal.
    if (banner.isSymmetric() && row != col) { entries.emplace_back(static_cast<IndexT>(col - 1), static_cast<IndexT>(row - 1), value); }
  }

  if (!dimensions_read) { throw std::runtime_error("Missing size line in MatrixMarket file"); }
  if (read_entries != nnz) {
    throw std::runtime_error("The MatrixMarket file has " + std::to_string(read_entries) + " entries, expected " + std::to_string(nnz));
  }

  // Sort entries by row, then by column
  std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
    return std::get<0>(a) < std::get<0>(b) || (std::get<0>(a) == std::get<0>(b) && std::get<1>(a) < std::get<1>(b));
  });

  // Initialize CSR vectors
  std::vector<OffsetT> row_offsets(rows + 1, 0);
  std::vector<IndexT> column_indices(entries.size());
  std::vector<ValueT> nnz_values(entries.size());

  // Count non-zero elements per row for row_offsets
  for (const auto& entry : entries) { row_offsets[std::get<0>(entry) + 1]++; }

  // Accumulate counts to get row offsets
  for (IndexT i = 1; i <= rows; ++i) { row_offsets[i] += row_offsets[i - 1]; }

  // Fill in column indices and values arrays
  std::vector<OffsetT> row_position(rows, 0);
  for (const auto& entry : entries) {
    IndexT row = std::get<0>(entry);
    IndexT col = std::get<1>(entry);
    ValueT value = std::get<2>(entry);

    OffsetT pos = row_offsets[row] + row_position[row];
    column_indices[pos] = col;
    nnz_values[pos] = value;
    row_position[row]++;
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Reads a Matrix Market file in coordinate format and converts it to a CSR matrix.
 *
 * This function parses a Matrix Market file from an input stream and converts it to a CSR (Compressed Sparse Row)
 * matrix format. The function only supports `coordinate` format and `general` or `symmetric` symmetry.
 * If the matrix is symmetric, both (row, col) and (col, row) entries are stored for each non-diagonal entry.
 *
 * @tparam ValueT Type of the non-zero values in the matrix.
 * @tparam IndexT Type of the indices (default is int).
 * @tparam OffsetT Type of the offsets (default is int).
 *
 * @param filename Name of the file containing the Matrix Market data.
 * @return CSR<ValueT, IndexT, OffsetT> An instance of the CSR class containing the matrix data in CSR format.
 *
 * @throws std::runtime_error if the Matrix Market format or symmetry type is unsupported.

 * @note The function expects the input to be in Matrix Market coordinate format.
 *       It does not support array format or field types other than real numbers.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromMM(const std::string& filename, sygraph::graph::Properties* properties = nullptr) {
  std::ifstream file(filename);
  if (!file.is_open()) { throw std::runtime_error("Failed to open file: " + filename); }

  return fromMM<ValueT, IndexT, OffsetT>(file, properties);
}

/**
 * @brief Converts a matrix in CSR format from a file to a CSR object.
 *
 * This function reads a matrix in CSR format from a given file and converts it into a CSR object.
 * The input holds the number of rows n, the n + 1 row offsets, then the column indices and the values of the
 * non-zero entries (as many as the last row offset).
 *
 * @tparam ValueT The value type of the matrix elements.
 * @tparam IndexT The index type used for column indices.
 * @tparam OffsetT The offset type used for row offsets.
 * @param fname The name of the file containing the matrix in CSR format.
 * @return The CSR object representing the matrix.
 * @throws std::runtime_error if the input ends early or a value cannot be parsed.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromCSR(std::istream& iss) {
  size_t n_rows = 0;
  if (!(iss >> n_rows)) { throw std::runtime_error("Truncated CSR input: missing row count"); }

  std::vector<OffsetT> row_offsets(n_rows + 1);
  for (auto& offset : row_offsets) {
    if (!(iss >> offset)) { throw std::runtime_error("Truncated CSR input: missing row offsets"); }
  }

  std::vector<IndexT> column_indices(row_offsets.back());
  for (auto& index : column_indices) {
    if (!(iss >> index)) { throw std::runtime_error("Truncated CSR input: missing column indices"); }
  }

  std::vector<ValueT> nnz_values(row_offsets.back());
  for (auto& value : nnz_values) {
    if (!(iss >> value)) { throw std::runtime_error("Truncated CSR input: missing values"); }
  }

  return sygraph::formats::CSR<ValueT, IndexT, OffsetT>(row_offsets, column_indices, nnz_values);
}

/**
 * @brief Converts a COO (Coordinate List) formatted sparse matrix to CSR (Compressed Sparse Row) format.
 *
 * @tparam ValueT The type of the values in the matrix.
 * @tparam IndexT The type of the indices in the matrix.
 * @tparam OffsetT The type of the offsets in the CSR format.
 * @param coo The input COO formatted sparse matrix.
 * @return A CSR formatted sparse matrix.
 *
 * This function takes a sparse matrix in COO format and converts it to CSR format.
 * The COO format stores the matrix as a list of (row, column, value) tuples,
 * while the CSR format uses three arrays: one for row offsets, one for column indices,
 * and one for values. The conversion involves counting the number of non-zero elements
 * in each row, computing prefix sums to determine row offsets, and then filling the
 * CSR arrays with the appropriate values and column indices.
 *
 * The CSR has `coo.getNumNodes()` rows, and the entries of each row are sorted by column.
 *
 * @throws std::runtime_error If an entry refers to a vertex outside the COO's vertex count.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromCOO(const sygraph::formats::COO<ValueT, IndexT, OffsetT>& coo) {
  const auto& coo_row_indices = coo.getRowIndices();
  const auto& coo_column_indices = coo.getColumnIndices();
  const auto& coo_values = coo.getValues();
  const size_t size = coo.getSize();
  const size_t n_nodes = coo.getNumNodes();

  // Count the number of nonzeros in each row.
  std::vector<OffsetT> csr_row_offsets(n_nodes + 1, 0);
  for (size_t i = 0; i < size; i++) {
    if (coo_row_indices[i] >= n_nodes || coo_column_indices[i] >= n_nodes) {
      throw std::runtime_error("COO entry out of range: (" + std::to_string(coo_row_indices[i]) + ", " + std::to_string(coo_column_indices[i])
                               + ") with " + std::to_string(n_nodes) + " vertices");
    }
    csr_row_offsets[coo_row_indices[i] + 1]++;
  }
  for (size_t i = 0; i < n_nodes; i++) { csr_row_offsets[i + 1] += csr_row_offsets[i]; }

  // Order the entries by row, then by column; equal entries keep their input order.
  std::vector<size_t> order(size);
  for (size_t i = 0; i < size; i++) { order[i] = i; }
  std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return coo_row_indices[a] != coo_row_indices[b] ? coo_row_indices[a] < coo_row_indices[b] : coo_column_indices[a] < coo_column_indices[b];
  });

  std::vector<IndexT> csr_column_indices(size);
  std::vector<ValueT> csr_values(size);
  for (size_t i = 0; i < size; i++) {
    csr_column_indices[i] = coo_column_indices[order[i]];
    csr_values[i] = coo_values[order[i]];
  }

  return {csr_row_offsets, csr_column_indices, csr_values};
}

/**
 * @brief Serializes a CSR (Compressed Sparse Row) matrix to a binary stream.
 *
 * This function writes the CSR matrix data to the provided output stream in binary format.
 * The CSR matrix is represented by its row offsets, column indices, and values arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row offsets in the CSR matrix.
 * @param csr The CSR matrix to be serialized.
 * @param oss The output stream to which the CSR matrix will be written.
 *
 * @throws std::runtime_error If the output stream is not in a good state.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
void toBinary(const sygraph::formats::CSR<ValueT, IndexT, OffsetT>& csr,
              std::ostream& oss,
              const sygraph::graph::Properties& properties = sygraph::graph::Properties()) {
  if (!oss) { throw std::runtime_error("Failed to write binary CSR matrix"); }

  auto& row_offsets = csr.getRowOffsets();
  auto& column_indices = csr.getColumnIndices();
  auto& values = csr.getValues();

  size_t num_rows = row_offsets.size();
  size_t num_nonzero = column_indices.size();

  uint64_t magic = detail::binary::magic;
  uint8_t version = 1;
  uint8_t flags = 0;
  if (properties.directed) { flags |= detail::binary::directed_mask; }
  if (properties.weighted) { flags |= detail::binary::weighted_mask; }
  uint16_t reserved16 = 0;
  uint32_t reserved32 = 0;

  oss.write(reinterpret_cast<const char*>(&magic), sizeof(uint64_t));
  oss.write(reinterpret_cast<const char*>(&version), sizeof(uint8_t));
  oss.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));
  oss.write(reinterpret_cast<const char*>(&reserved16), sizeof(uint16_t));
  oss.write(reinterpret_cast<const char*>(&reserved32), sizeof(uint32_t));

  oss.write(reinterpret_cast<const char*>(&num_rows), sizeof(size_t));
  oss.write(reinterpret_cast<const char*>(&num_nonzero), sizeof(size_t));

  oss.write(reinterpret_cast<const char*>(row_offsets.data()), row_offsets.size() * sizeof(OffsetT));
  oss.write(reinterpret_cast<const char*>(column_indices.data()), column_indices.size() * sizeof(IndexT));
  oss.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(ValueT));
}

/**
 * @brief Reads a CSR (Compressed Sparse Row) matrix from a binary input stream.
 *
 * This function reads the number of rows and non-zero elements from the binary
 * input stream, followed by the row pointers, column indices, and values arrays.
 * It then constructs and returns a CSR matrix using these arrays.
 *
 * @tparam ValueT The type of the values in the CSR matrix.
 * @tparam IndexT The type of the column indices in the CSR matrix.
 * @tparam OffsetT The type of the row pointers in the CSR matrix.
 * @param iss The input stream to read the binary data from.
 * @return A CSR matrix containing the data read from the input stream.
 * @throws std::runtime_error If the input stream is not valid, the data ends early, or the row offsets are inconsistent
 * with the number of non-zero entries.
 */
template<typename ValueT, typename IndexT, typename OffsetT>
sygraph::formats::CSR<ValueT, IndexT, OffsetT> fromBinary(std::istream& iss, sygraph::graph::Properties* properties = nullptr) {
  if (!iss) { throw std::runtime_error("Failed to read binary CSR matrix"); }

  size_t num_rows = 0;
  size_t num_nonzero = 0;
  uint64_t maybe_magic = 0;

  iss.read(reinterpret_cast<char*>(&maybe_magic), sizeof(uint64_t));
  if (!iss) { throw std::runtime_error("Failed to read binary CSR matrix"); }

  if (maybe_magic == detail::binary::magic) {
    uint8_t version = 0;
    uint8_t flags = 0;
    uint16_t reserved16 = 0;
    uint32_t reserved32 = 0;

    iss.read(reinterpret_cast<char*>(&version), sizeof(uint8_t));
    iss.read(reinterpret_cast<char*>(&flags), sizeof(uint8_t));
    iss.read(reinterpret_cast<char*>(&reserved16), sizeof(uint16_t));
    iss.read(reinterpret_cast<char*>(&reserved32), sizeof(uint32_t));
    iss.read(reinterpret_cast<char*>(&num_rows), sizeof(size_t));
    iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));
    if (!iss) { throw std::runtime_error("Truncated binary CSR matrix: incomplete header"); }

    if (properties) {
      properties->directed = (flags & detail::binary::directed_mask) != 0;
      properties->weighted = (flags & detail::binary::weighted_mask) != 0;
    }
  } else {
    num_rows = static_cast<size_t>(maybe_magic);
    iss.read(reinterpret_cast<char*>(&num_nonzero), sizeof(size_t));
    if (!iss) { throw std::runtime_error("Truncated binary CSR matrix: incomplete header"); }
    if (properties) {
      properties->directed = true;
      properties->weighted = true;
    }
  }

  std::vector<OffsetT> row_ptr(num_rows);
  std::vector<IndexT> col_indices(num_nonzero);
  std::vector<ValueT> values(num_nonzero);

  iss.read(reinterpret_cast<char*>(row_ptr.data()), row_ptr.size() * sizeof(OffsetT));
  iss.read(reinterpret_cast<char*>(col_indices.data()), col_indices.size() * sizeof(IndexT));
  iss.read(reinterpret_cast<char*>(values.data()), values.size() * sizeof(ValueT));
  if (!iss) { throw std::runtime_error("Truncated binary CSR matrix: incomplete data"); }

  // The offsets must start at 0, never decrease, and end at the number of non-zero entries.
  bool consistent = num_rows > 0 && row_ptr.front() == 0 && static_cast<size_t>(row_ptr.back()) == num_nonzero;
  for (size_t i = 1; consistent && i < num_rows; i++) { consistent = row_ptr[i - 1] <= row_ptr[i]; }
  if (!consistent) { throw std::runtime_error("Invalid binary CSR matrix: row offsets do not match the number of non-zero entries"); }

  return {row_ptr, col_indices, values};
}
} // namespace csr
} // namespace io
} // namespace sygraph
