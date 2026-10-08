#include "test_utils.hpp"

#include <sstream>
#include <string>
#include <vector>

using sygraph::graph::Properties;
using sygraph::tests::expectThrows;

template<typename ValueT>
using csr_t = sygraph::formats::CSR<ValueT, uint, uint>;

template<typename ValueT>
void expectSame(const csr_t<ValueT>& a, const csr_t<ValueT>& b) {
  assert(a.getRowOffsets() == b.getRowOffsets());
  assert(a.getColumnIndices() == b.getColumnIndices());
  assert(a.getValues() == b.getValues());
}

template<typename ValueT>
std::string toBinary(const csr_t<ValueT>& csr, const Properties& properties) {
  std::ostringstream out(std::ios::binary);
  sygraph::io::csr::toBinary(csr, out, properties);
  return out.str();
}

template<typename ValueT>
csr_t<ValueT> fromBinary(const std::string& bytes, Properties* properties = nullptr) {
  std::istringstream in(bytes, std::ios::binary);
  return sygraph::io::csr::fromBinary<ValueT, uint, uint>(in, properties);
}

template<typename ValueT>
void expectRoundTrip(const csr_t<ValueT>& csr) {
  for (Properties properties : {Properties{false, false}, Properties{true, false}, Properties{false, true}, Properties{true, true}}) {
    Properties read_properties{!properties.directed, !properties.weighted};
    expectSame(fromBinary<ValueT>(toBinary(csr, properties), &read_properties), csr);
    assert(read_properties.directed == properties.directed);
    assert(read_properties.weighted == properties.weighted);
  }
}

template<typename T>
void writeRaw(std::string& bytes, const T& value) {
  bytes.append(reinterpret_cast<const char*>(&value), sizeof(T));
}

int main() {
  // Binary round trips keep the exact arrays and the properties.
  expectRoundTrip(sygraph::tests::gen::randomGraph(500, 2000, /*seed=*/73, /*directed=*/true, /*max_weight=*/50));
  expectRoundTrip(csr_t<float>{{0, 2, 2, 3}, {1, 2, 0}, {0.25f, 1.5f, -3.0f}});
  expectRoundTrip(csr_t<uint>{{0, 0, 0, 0}, {}, {}}); // vertices but no edges
  expectRoundTrip(csr_t<uint>{{0}, {}, {}});          // no vertices

  // Files written before the header was introduced (row count, edge count, arrays) are still readable.
  const csr_t<uint> legacy_csr{{0, 1, 2}, {1, 0}, {5, 6}};
  std::string legacy;
  writeRaw(legacy, size_t{3});
  writeRaw(legacy, size_t{2});
  for (uint v : legacy_csr.getRowOffsets()) { writeRaw(legacy, v); }
  for (uint v : legacy_csr.getColumnIndices()) { writeRaw(legacy, v); }
  for (uint v : legacy_csr.getValues()) { writeRaw(legacy, v); }
  Properties legacy_properties;
  expectSame(fromBinary<uint>(legacy, &legacy_properties), legacy_csr);
  assert(legacy_properties.directed && legacy_properties.weighted);

  // Truncated or inconsistent binary data is rejected.
  const auto bytes = toBinary(legacy_csr, Properties{});
  expectThrows([] { fromBinary<uint>(""); }, "Failed to read binary CSR");
  expectThrows([&] { fromBinary<uint>(bytes.substr(0, bytes.size() - 1)); }, "Truncated");
  expectThrows([&] { fromBinary<uint>(bytes.substr(0, 20)); }, "Truncated");
  std::string bad_offsets = bytes;
  bad_offsets[32 + 2 * sizeof(uint)] = 7; // last row offset no longer matches the edge count
  expectThrows([&] { fromBinary<uint>(bad_offsets); }, "Invalid binary CSR");

  // Text CSR: row count, the n + 1 row offsets, the column indices, then the values.
  std::istringstream text("3\n0 1 3 4\n1 0 2 1\n5 6 7 8\n");
  auto from_text = sygraph::io::csr::fromCSR<uint, uint, uint>(text);
  expectSame(from_text, csr_t<uint>{{0, 1, 3, 4}, {1, 0, 2, 1}, {5, 6, 7, 8}});
  assert(from_text.getRowOffsetsSize() == 3);
  expectThrows(
      [] {
        std::istringstream truncated("3\n0 1 3 4\n1 0 2\n");
        sygraph::io::csr::fromCSR<uint, uint, uint>(truncated);
      },
      "Truncated");
}
