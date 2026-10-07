#include "test_utils.hpp"
#include <sycl/sycl.hpp>
#include <sygraph/sygraph.hpp>

constexpr size_t NUM_ELEMS = 128;

int main() {
  auto q = sygraph::tests::makeQueue();

  sygraph::frontier::Frontier<size_t> f{q, NUM_ELEMS};
  auto n = f.size();
  using type = typename sygraph::frontier::Frontier<size_t>::bitmap_type;
  assert(n == 0);
  assert(sizeof(type) * sygraph::types::detail::byte_size == f.getBitmapRange());
  assert(f.getNumElems() == NUM_ELEMS);

  q.submit([&](sycl::handler& cgh) {
     auto bitmap = f.getDeviceFrontier();
     cgh.parallel_for(sycl::range<1>{f.getNumElems()}, [=](sycl::id<1> idx) { bitmap.insert(idx); });
   }).wait();

  n = f.size();
  assert(n == NUM_ELEMS);

  f.remove(0);
  n = f.size();
  assert(n == NUM_ELEMS - 1);
}
