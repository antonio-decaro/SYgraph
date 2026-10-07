#include "test_utils.hpp"
#include <iostream>
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  auto q = sygraph::tests::makeQueue();

  std::string mat = "4 4 8\n"
                    "1 2\n"
                    "1 0\n"
                    "0 2\n"
                    "2 0\n"
                    "1 3\n"
                    "2 1\n"
                    "3 1\n"
                    "0 1";
  std::istringstream iss(mat.data());
  auto coo = sygraph::io::coo::fromCOO<uint, uint, uint>(iss);

  auto csr = sygraph::io::csr::fromCOO(coo);
  auto row_offsets = csr.getRowOffsets();
  auto col_indices = csr.getColumnIndices();
  auto values = csr.getValues();

  // Each row is sorted by column; edges without an explicit weight get weight 1.
  assert(row_offsets == (std::vector<uint>{0, 2, 5, 7, 8}));
  assert(col_indices == (std::vector<uint>{1, 2, 0, 2, 3, 0, 1, 1}));
  assert(values == std::vector<uint>(8, 1));
}
