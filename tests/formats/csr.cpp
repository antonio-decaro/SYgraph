#include "test_utils.hpp"
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  auto q = sygraph::tests::makeQueue();

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);

  // Adjacency: 0:{1,2} 1:{0,2} 2:{0,1,3,4} 3:{2} 4:{2,5} 5:{4}
  assert(csr.getRowOffsetsSize() == 6);
  assert(csr.getRowOffsets() == (std::vector<uint>{0, 2, 4, 8, 9, 11, 12}));
  assert(csr.getColumnIndices() == (std::vector<uint>{1, 2, 0, 2, 0, 1, 3, 4, 2, 2, 5, 4}));
  assert(csr.getValues() == std::vector<uint>(12, 1));

  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);
  assert(G.getVertexCount() == csr.getRowOffsets().size() - 1);
  assert(G.getEdgeCount() == csr.getNumNonzeros());
  for (size_t e = 0; e < G.getValuesSize(); ++e) { assert(G.getValues()[e] == csr.getValues()[e]); }

  // invert() turns in-edges into out-edges and keeps the weights:
  // 0->1 (1), 0->2 (4), 1->2 (2), 1->3 (6), 2->3 (1), 2->4 (5), 3->4 (1).
  std::istringstream directed_iss{std::string(sygraph::tests::fixtures::weighted_directed_5)};
  auto directed = sygraph::io::csr::fromMatrix<uint, uint, uint>(directed_iss);
  auto inverted = directed.invert();
  assert(inverted.getRowOffsets() == (std::vector<uint>{0, 0, 1, 3, 5, 7}));
  assert(inverted.getColumnIndices() == (std::vector<uint>{0, 0, 1, 1, 2, 2, 3}));
  assert(inverted.getValues() == (std::vector<uint>{1, 4, 2, 6, 1, 5, 1}));
}
