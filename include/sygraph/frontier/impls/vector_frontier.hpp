#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/utils/vector.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template<typename type_t>
class frontier_vector_t;
template<typename type_t>
class device_vector_frontier_t;

template<typename type_t>
class device_vector_frontier_t {
  friend class frontier_vector_t<type_t>;

public:
  device_vector_frontier_t(size_t max_size) : max_size(max_size) {}

  SYCL_EXTERNAL inline bool empty() const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load() == 0;
  }

  SYCL_EXTERNAL inline size_t size() const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load();
  }

  SYCL_EXTERNAL inline bool insert(type_t val) const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    if (ref.load() >= max_size) { return false; }
    data[ref++] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool insert(type_t val, size_t idx) const {
    if (idx >= *tail) { return false; }
    data[idx] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool remove(type_t val) const { return false; }

  SYCL_EXTERNAL inline void clear() const {
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    ref = 0;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { // TODO: [!] check for max size
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.fetch_add(num_elems);
  }

  template<typename T, sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool
  finalize(sycl::nd_item<1> item, const sycl::local_accessor<type_t, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail_ref) const {
    auto group = item.get_group();
    auto lid = item.get_local_linear_id();
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> tail_ref(tail[0]);

    size_t data_offset = max_size;
    if (group.leader()) { data_offset = tail_ref.fetch_add(pad_tail_ref.load()); }
    data_offset = sycl::group_broadcast(group, data_offset, 0);
    for (int i = lid; i < pad_tail_ref.load() && i < max_size; i += item.get_local_range(0)) { data[data_offset + i] = pad[i]; }

    return true;
  }

  SYCL_EXTERNAL size_t getVectorMaxSize() const { return max_size; }

  SYCL_EXTERNAL uint32_t getVectorSize() const { return *tail; }

  SYCL_EXTERNAL uint32_t* getVectorSizePtr() const { return tail; }

  template<typename T, sycl::memory_order MO, sycl::memory_scope MS>
  SYCL_EXTERNAL inline bool insert(type_t val, const sycl::local_accessor<type_t, 1>& pad, const sycl::atomic_ref<T, MO, MS>& pad_tail) const {
    if (pad_tail.load() < max_size) {
      pad[pad_tail++] = val;
    } else {
      return insert(val);
    }
    return true;
  }

protected:
  inline void set_tail(uint32_t new_tail) { *tail = new_tail; }
  inline size_t get_tail() const { return *tail; }

private:
  type_t* data;
  uint32_t* tail;
  size_t max_size;
};

template<typename type_t>
class frontier_vector_t {
public:
  using frontier_type = type_t;

  static void swap(frontier_vector_t<type_t>& a, frontier_vector_t<type_t>& b) {
    std::swap(a.vector.data, b.vector.data);
    std::swap(a.vector.tail, b.vector.tail);
  }

  frontier_vector_t(sycl::queue& q, size_t num_elems) : q(q), vector(num_elems) {
    vector.data = memory::detail::memoryAlloc<type_t, memory::space::device>(num_elems, q);
    vector.tail = memory::detail::memoryAlloc<uint32_t, memory::space::device>(1, q);
    q.fill(vector.tail, 0, 1).wait();
  }

  ~frontier_vector_t() {
    sycl::free(vector.data, q);
    sycl::free(vector.tail, q);
  }

  size_t getNumActiveElements() const { return vector.get_tail(); }

  inline bool selfAllocated() const { return true; }

  void getActiveElements(type_t*& elems) const { elems = vector.data; }

  /**
   * @brief Retrieves the active elements in the bitmap.
   *
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
   */
  void getActiveElements(type_t*& elems, size_t& size) const {
    size = this->getVectorSize();
    elems = vector.data;
  }

  inline bool empty() const { return this->getVectorSize() == 0; }

  bool insert(type_t val) {
    q.submit([&](sycl::handler& cgh) { cgh.single_task([=, data = vector.data, tail = vector.tail]() { data[(*(tail))++] = val; }); }).wait();
    return true;
  }

  bool insert(type_t val, size_t idx) {
    if (idx >= this->getVectorSize()) { return false; }
    q.submit([&](sycl::handler& cgh) { cgh.single_task([=, data = vector.data]() { data[idx] = val; }); }).wait();
    return true;
  }

  bool remove(type_t idx) { return false; }

  frontier_vector_t& operator=(const frontier_vector_t& other) { throw std::runtime_error("Not implemented"); }

  inline void merge(frontier_bitmap_t<type_t>& other) { throw std::runtime_error("Not implemented"); }

  inline void clear() { q.fill(vector.tail, 0, 1).wait(); }

  inline type_t* getVector() const { return vector.data; }

  inline size_t getVectorSize() const {
    uint32_t size;
    q.copy(vector.tail, &size, 1).wait();
    return size;
  }

  const device_vector_frontier_t<type_t>& getDeviceFrontier() const { return vector; }

private:
  sycl::queue& q; ///< The SYCL queue used for memory allocation.
  device_vector_frontier_t<type_t> vector;
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph