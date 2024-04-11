#pragma once
#include <sycl/sycl.hpp>
#include <memory>
#include <vector>

#include <sygraph/utils/types.hpp>
#include <sygraph/utils/memory.hpp>

namespace sygraph {
inline namespace v0 {
namespace formats {


template <typename value_t,
          typename index_t = types::index_t, 
          typename offset_t = types::offset_t>
class CSR {
public:
  CSR(index_t n_rows, offset_t n_nonzeros) {
    row_offsets.resize(n_rows + 1);
    column_indices.resize(n_nonzeros);
    nnz_values.resize(n_nonzeros);  
  }

  ~CSR() = default;

  // Getters
  index_t getNumRows() const {
    return row_offsets.size() - 1;
  }

  offset_t getNumNonzeros() const {
    return column_indices.size();
  }

  const std::vector<offset_t>& getRowOffsets() const {
    return row_offsets;
  }
  std::vector<offset_t>& getRowOffsets() {
    return row_offsets;
  }

  const std::vector<index_t>& getColumnIndices() const {
    return column_indices;
  }
  std::vector<index_t>& getColumnIndices() {
    return column_indices;
  }

  const std::vector<value_t>& getNnzValues() const {
    return nnz_values;
  }
  std::vector<value_t>& getNnzValues() {
    return nnz_values;
  }

  void setRowOffsets(const std::vector<offset_t>& offsets) {
    row_offsets = offsets;
  }

  void setColumnIndices(const std::vector<index_t>& indices) {
    column_indices = indices;
  }

  void setNnzValues(const std::vector<value_t>& values) {
    nnz_values = values;
  }

private:

  std::vector<offset_t> row_offsets;
  std::vector<index_t> column_indices;
  std::vector<value_t> nnz_values;  
};

} // namespace formats
} // namespace v0
} // namespace sygraph