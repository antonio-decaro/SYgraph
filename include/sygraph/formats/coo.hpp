/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <algorithm>
#include <memory>
#include <sycl/sycl.hpp>
#include <vector>

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
namespace formats {


/**
 * @class COO
 * @brief A class representing a sparse matrix in Coordinate (COO) format.
 *
 * This class provides methods to store and manipulate sparse matrices using
 * the Coordinate (COO) format, which stores the row indices, column indices,
 * and non-zero values of the matrix.
 *
 * @tparam ValueT The type of the non-zero values.
 * @tparam IndexT The type of the row and column indices.
 * @tparam OffsetT The type of the row offsets.
 *
 * @note The class assumes that the input vectors for row indices, column indices,
 * and non-zero values are of the same length.
 *
 * @section Example
 * @code
 * std::vector<int> row_indices = {0, 1, 2};
 * std::vector<int> column_indices = {0, 1, 2};
 * std::vector<double> values = {1.0, 2.0, 3.0};
 * COO<int, int, double> coo_matrix(row_indices, column_indices, values);
 * @endcode
 */
template<typename ValueT, typename IndexT = types::index_t, typename OffsetT = types::offset_t>
class COO {
public:
  using value_t = ValueT;
  using index_t = IndexT;
  using offset_t = OffsetT;

  /**
   * @brief Constructs a COO (Coordinate Format) object.
   *
   * @param row_indices A vector containing the row indices of the non-zero elements.
   * @param column_indices A vector containing the column indices of the non-zero elements.
   * @param nnz_values A vector containing the values of the non-zero elements.
   * @param num_nodes The number of vertices. If omitted, it is the largest index plus one (0 when there are no entries).
   */
  COO(std::vector<index_t> row_indices, std::vector<index_t> column_indices, std::vector<value_t> nnz_values, size_t num_nodes = 0)
      : _row_indices(row_indices), _column_indices(column_indices), _nnz_values(nnz_values) {
    _num_nodes = num_nodes > 0 ? num_nodes : derivedNumNodes();
  }

  /**
   * @brief Constructs a COO (Coordinate Format) object with pre-allocated space for the given number of values.
   *
   * This constructor initializes the COO object by reserving space in the internal vectors
   * for row indices, column indices, and non-zero values based on the specified number of values.
   *
   * @param values The number of non-zero values to reserve space for in the COO format.
   */
  COO(size_t values) {
    _row_indices.reserve(values);
    _column_indices.reserve(values);
    _nnz_values.reserve(values);
  }

  ~COO() = default;

  /**
   * @brief Retrieves the row indices of the COO (Coordinate) format matrix.
   *
   * This function returns a constant reference to a vector containing the row indices
   * of the non-zero elements in the matrix. The row indices are stored in a vector
   * of type `index_t`.
   *
   * @return const std::vector<index_t>& A constant reference to the vector of row indices.
   */
  const std::vector<index_t>& getRowIndices() const { return _row_indices; }


  /**
   * @brief Retrieves the column indices of the COO (Coordinate List) format.
   *
   * This function returns a constant reference to a vector containing the column
   * indices of the non-zero elements in the sparse matrix represented in COO format.
   *
   * @return const std::vector<index_t>& A constant reference to the vector of column indices.
   */
  const std::vector<index_t>& getColumnIndices() const { return _column_indices; }

  /**
   * @brief Retrieves the non-zero values of the COO (Coordinate List) format.
   *
   * This function returns a constant reference to a vector containing the non-zero
   * values of the sparse matrix represented in COO format.
   *
   * @return const std::vector<value_t>& A constant reference to the vector of non-zero values.
   */
  const std::vector<value_t>& getValues() const { return _nnz_values; }

  /**
   * @brief Retrieves the row indices of the COO (Coordinate List) format.
   *
   * This function returns a reference to the vector containing the row indices
   * of the sparse matrix stored in COO format. The row indices indicate the
   * row positions of the non-zero elements in the matrix.
   *
   * @return std::vector<index_t>& Reference to the vector of row indices.
   */
  std::vector<index_t>& getRowIndices() { return _row_indices; }

  /**
   * @brief Retrieves the column indices of the COO (Coordinate) format matrix.
   *
   * This function returns a reference to the vector containing the column indices
   * of the non-zero elements in the matrix. The column indices correspond to the
   * positions of the non-zero elements in the matrix.
   *
   * @return std::vector<index_t>& Reference to the vector of column indices.
   */
  std::vector<index_t>& getColumnIndices() { return _column_indices; }


  /**
   * @brief Retrieves the non-zero values of the sparse matrix in COO format.
   *
   * This function returns a reference to a vector containing the non-zero values
   * (NNZ) of the sparse matrix stored in Coordinate (COO) format.
   *
   * @return A reference to a vector of non-zero values.
   */
  std::vector<value_t>& getValues() { return _nnz_values; }


  /**
   * @brief Get the size of the COO format data.
   *
   * This function returns the number of elements in the COO format
   * by returning the size of the row indices vector.
   *
   * @return size_t The number of elements in the COO format.
   */
  size_t getSize() const { return _row_indices.size(); }

  /**
   * @brief Returns the number of vertices, which can exceed the largest index when the last vertices have no edges.
   */
  size_t getNumNodes() const { return _num_nodes; }

  void setNumNodes(size_t num_nodes) { _num_nodes = num_nodes; }

private:
  size_t derivedNumNodes() const {
    if (_row_indices.empty()) { return 0; }
    return static_cast<size_t>(std::max(*std::max_element(_row_indices.begin(), _row_indices.end()),
                                        *std::max_element(_column_indices.begin(), _column_indices.end())))
           + 1;
  }

  size_t _num_nodes = 0;
  std::vector<index_t> _row_indices;
  std::vector<index_t> _column_indices;
  std::vector<value_t> _nnz_values;
};

} // namespace formats
} // namespace sygraph