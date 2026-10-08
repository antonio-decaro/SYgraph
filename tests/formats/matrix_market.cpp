#include "test_utils.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using sygraph::graph::Properties;
using sygraph::tests::expectThrows;

template<typename ValueT>
auto readMM(const std::string& text, Properties* properties = nullptr) {
  std::istringstream stream(text);
  return sygraph::io::csr::fromMM<ValueT, uint, uint>(stream, properties);
}

template<typename CsrT, typename ValueT>
void expectCSR(const CsrT& csr, const std::vector<uint>& offsets, const std::vector<uint>& columns, const std::vector<ValueT>& values) {
  assert(csr.getRowOffsets() == offsets);
  assert(csr.getColumnIndices() == columns);
  assert(csr.getValues() == values);
}

int main() {
  // General real matrix: entries are sorted by row and column, values kept.
  Properties props;
  auto general = readMM<float>("%%MatrixMarket matrix coordinate real general\n"
                               "% a comment\n"
                               "3 3 3\n"
                               "3 1 2.5\n"
                               "1 2 1.5\n"
                               "2 3 0.5\n",
                               &props);
  expectCSR(general, {0, 1, 2, 3}, {1, 2, 0}, std::vector<float>{1.5f, 0.5f, 2.5f});
  assert(props.directed && props.weighted);

  // Integer values into an integral type.
  auto integer = readMM<uint>("%%MatrixMarket matrix coordinate integer general\n"
                              "2 2 2\n"
                              "1 2 7\n"
                              "2 1 9\n");
  expectCSR(integer, {0, 1, 2}, {1, 0}, std::vector<uint>{7, 9});

  // Symmetric pattern: off-diagonal entries are mirrored, diagonal entries are not duplicated.
  auto symmetric = readMM<uint>("%%MatrixMarket matrix coordinate pattern symmetric\n"
                                "3 3 3\n"
                                "2 1\n"
                                "3 2\n"
                                "2 2\n",
                                &props);
  expectCSR(symmetric, {0, 1, 4, 5}, {1, 0, 1, 2, 1}, std::vector<uint>{1, 1, 1, 1, 1});
  assert(!props.directed && !props.weighted);

  // Keywords are case-insensitive; blank lines and comments between entries are skipped.
  auto relaxed = readMM<uint>("%%MatrixMarket Matrix Coordinate Pattern General\n"
                              "%\n"
                              "\n"
                              "2 2 2\n"
                              "1 2\n"
                              "\n"
                              "% between entries\n"
                              "2 1\n"
                              "\n");
  expectCSR(relaxed, {0, 1, 2}, {1, 0}, std::vector<uint>{1, 1});

  // An isolated last vertex keeps its (empty) row.
  auto isolated = readMM<uint>("%%MatrixMarket matrix coordinate pattern general\n4 4 1\n1 2\n");
  expectCSR(isolated, {0, 1, 1, 1, 1}, {1}, std::vector<uint>{1});

  // Reading from a file.
  const auto path = std::filesystem::temp_directory_path() / "sygraph_matrix_market_test.mtx";
  {
    std::ofstream file(path);
    file << "%%MatrixMarket matrix coordinate pattern general\n2 2 1\n2 1\n";
  }
  auto from_file = sygraph::io::csr::fromMM<uint, uint, uint>(path.string());
  expectCSR(from_file, {0, 0, 1}, {0}, std::vector<uint>{1});
  std::filesystem::remove(path);
  expectThrows([&] { sygraph::io::csr::fromMM<uint, uint, uint>(path.string()); }, "Failed to open file");

  // Malformed or unsupported input is rejected with a clear error.
  const std::string pattern = "%%MatrixMarket matrix coordinate pattern general\n";
  expectThrows([] { readMM<uint>("3 3 1\n1 2\n"); }, "Missing MatrixMarket banner");
  expectThrows([] { readMM<uint>("%%NotMatrixMarket matrix coordinate pattern general\n1 1 0\n"); }, "Invalid MatrixMarket banner");
  expectThrows([] { readMM<uint>("%%MatrixMarket matrix coordinate pattern sideways\n1 1 0\n"); }, "Invalid symmetry");
  expectThrows([] { readMM<uint>("%%MatrixMarket matrix array real general\n1 1\n1\n"); }, "Unsupported MatrixMarket format");
  expectThrows([] { readMM<uint>("%%MatrixMarket vector coordinate real general\n1 1 0\n"); }, "Unsupported MatrixMarket object");
  expectThrows([] { readMM<float>("%%MatrixMarket matrix coordinate complex general\n1 1 0\n"); }, "Unsupported MatrixMarket field");
  expectThrows([] { readMM<float>("%%MatrixMarket matrix coordinate real skew-symmetric\n1 1 0\n"); }, "Unsupported MatrixMarket symmetry");
  expectThrows([] { readMM<float>("%%MatrixMarket matrix coordinate real hermitian\n1 1 0\n"); }, "Unsupported MatrixMarket symmetry");
  expectThrows([] { readMM<uint>("%%MatrixMarket matrix coordinate real general\n1 1 0\n"); }, "Invalid MatrixMarket field type");
  expectThrows([&] { readMM<uint>(pattern); }, "Missing size line");
  expectThrows([&] { readMM<uint>(pattern + "3 4 1\n1 2\n"); }, "square");
  expectThrows([&] { readMM<uint>(pattern + "3 3 1\n1 4\n"); }, "out of range");
  expectThrows([&] { readMM<uint>(pattern + "3 3 1\n0 1\n"); }, "out of range");
  expectThrows([&] { readMM<uint>(pattern + "3 3 2\n1 2\n"); }, "expected 2");
  expectThrows([&] { readMM<uint>(pattern + "3 3 1\n1 2\n2 3\n"); }, "expected 1");
  expectThrows([&] { readMM<uint>(pattern + "3 3 1\n1 x\n"); }, "Malformed");
  expectThrows([] { readMM<uint>("%%MatrixMarket matrix coordinate integer general\n3 3 1\n1 2\n"); }, "Malformed");
}
