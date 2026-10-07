#include "test_utils.hpp"

#include <string>
#include <vector>

using sygraph::graph::Properties;
using sygraph::tests::expectThrows;

template<typename ValueT = uint, typename OffsetT = uint>
auto readCOO(const std::string& text, bool undirected = false, Properties* properties = nullptr) {
  std::istringstream stream(text);
  return sygraph::io::coo::fromCOO<ValueT, uint, OffsetT>(stream, undirected, properties);
}

template<typename CsrT, typename OffsetT, typename ValueT>
void expectCSR(const CsrT& csr, const std::vector<OffsetT>& offsets, const std::vector<uint>& columns, const std::vector<ValueT>& values) {
  assert(csr.getRowOffsets() == offsets);
  assert(csr.getColumnIndices() == columns);
  assert(csr.getValues() == values);
}

int main() {
  // Directed, weighted: rows are sorted by column and weights follow their edges.
  Properties props;
  auto weighted = readCOO("% comment\n"
                          "5 5 3\n"
                          "0 4 1\n"
                          "3 1 4\n"
                          "0 1 2\n",
                          false,
                          &props);
  assert(weighted.getNumNodes() == 5);
  assert(props.directed && props.weighted);
  expectCSR(sygraph::io::csr::fromCOO(weighted), std::vector<uint>{0, 2, 2, 2, 3, 3}, {1, 4, 1}, std::vector<uint>{2, 1, 4});

  // Undirected, unweighted: every edge is stored in both directions with weight 1.
  auto undirected = readCOO("3 3 2\n0 1\n1 2\n", true, &props);
  assert(!props.directed && !props.weighted);
  assert(undirected.getSize() == 4);
  expectCSR(sygraph::io::csr::fromCOO(undirected), std::vector<uint>{0, 1, 3, 4}, {1, 0, 2, 1}, std::vector<uint>{1, 1, 1, 1});

  // The header's vertex count keeps trailing isolated vertices; blank lines and comments are skipped.
  auto isolated = readCOO("6 6 1\n\n% edge list\n0 1\n\n");
  expectCSR(sygraph::io::csr::fromCOO(isolated), std::vector<uint>{0, 1, 1, 1, 1, 1, 1}, {1}, std::vector<uint>{1});

  // An empty edge list.
  auto empty = readCOO("4 4 0\n");
  expectCSR(sygraph::io::csr::fromCOO(empty), std::vector<uint>{0, 0, 0, 0, 0}, {}, std::vector<uint>{});

  // Index and offset types may differ.
  auto wide = readCOO<float, size_t>("3 3 2\n2 0 1.5\n0 2 0.5\n");
  expectCSR(sygraph::io::csr::fromCOO(wide), std::vector<size_t>{0, 1, 1, 2}, {2, 0}, std::vector<float>{0.5f, 1.5f});

  // A COO built in code without a vertex count uses the largest index; an empty one has no vertices.
  sygraph::formats::COO<uint, uint, uint> built({2, 0}, {0, 3}, {1, 1});
  expectCSR(sygraph::io::csr::fromCOO(built), std::vector<uint>{0, 1, 1, 2, 2}, {3, 0}, std::vector<uint>{1, 1});
  sygraph::formats::COO<uint, uint, uint> none({}, {}, {});
  expectCSR(sygraph::io::csr::fromCOO(none), std::vector<uint>{0}, {}, std::vector<uint>{});

  // Malformed input is rejected with a clear error.
  expectThrows([] { readCOO(""); }, "could not read");
  expectThrows([] { readCOO("3 4 1\n0 1\n"); }, "square");
  expectThrows([] { readCOO("3 3 1\n0 1\n1 2\n"); }, "expected 1");
  expectThrows([] { readCOO("3 3 2\n0 1\n"); }, "expected 2");
  expectThrows([] { readCOO("3 3 1\n0 3\n"); }, "out of range");
  expectThrows([] { readCOO("3 3 1\n0 a\n"); }, "Malformed");
  expectThrows([] { readCOO("3 3\n"); }, "Malformed header");
}
