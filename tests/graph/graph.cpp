#include "test_utils.hpp"
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

int main() {
  auto q = sygraph::tests::makeQueue();

  auto mat = sygraph::io::storage::matrices::symmetric_6nodes;
  std::istringstream iss(mat.data());
  auto csr = sygraph::io::csr::fromMatrix<uint, uint, uint>(iss);
  auto G = sygraph::graph::build::fromCSR<sygraph::memory::space::shared>(q, csr);

  assert(G.getVertexCount() == csr.getRowOffsetsSize());
  assert(G.getEdgeCount() == csr.getNumNonzeros());

  using type_t = decltype(G)::vertex_t;
  sycl::buffer<type_t> nighbours_count_buf(G.getVertexCount());
  sycl::buffer<type_t> first_neighbors_buf(G.getVertexCount());

  q.submit([&](sycl::handler& h) {
    auto v = G.getValues();
    auto dg = G.getDeviceGraph();
    sycl::accessor nighbours_count{nighbours_count_buf, h, sycl::write_only};
    sycl::accessor first_neighbors{first_neighbors_buf, h, sycl::write_only};

    h.parallel_for(sycl::range<1>{G.getVertexCount()}, [=](sycl::id<1> idx) {
      auto id = idx[0];
      nighbours_count[id] = dg.getDegree(id);
      first_neighbors[id] = dg.getFirstNeighbor(id);
    });
  });
  q.wait();

  sycl::host_accessor nighbours_count{nighbours_count_buf};
  sycl::host_accessor first_neighbors{first_neighbors_buf};

  for (type_t i = 0; i < G.getVertexCount(); i++) {
    assert(nighbours_count[i] == csr.getRowOffsets()[i + 1] - csr.getRowOffsets()[i]);
    assert(first_neighbors[i] == csr.getRowOffsets()[i]);
  }
}
