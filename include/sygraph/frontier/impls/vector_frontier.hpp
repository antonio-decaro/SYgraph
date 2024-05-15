#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template <typename type_t>
class frontier_vector_t;
template <typename type_t>
class device_vector_frontier_t;

template <typename type_t>
struct local_vector_context_t {

  local_vector_context_t(size_t max_elems, sycl::handler& cgh) : max_elems(max_elems), data(max_elems, cgh), tail(1, cgh) {} // TODO: [!!] fix this

  SYCL_EXTERNAL static inline void init(const local_vector_context_t& context, sycl::nd_item<1>& item) {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space> ref (context.tail[0]);
    if (item.get_local_linear_id() == 0) {
      ref = 0;
    }
    sycl::group_barrier(item.get_group());
  }

  SYCL_EXTERNAL static inline bool insert(const local_vector_context_t& context, type_t val) {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref (context.tail[0]);
    // size_t& ref = tail[0];
    if (ref >= context.max_elems) {
      return false;
    }
    size_t loc = ref.fetch_add(1);
    context.data[loc] = val;
    return true;
  }

  SYCL_EXTERNAL static inline void copy_to_global(const local_vector_context_t& context, sycl::nd_item<1>& item, const device_vector_frontier_t<type_t>& out) {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::local_space> ref (context.tail[0]);
    size_t lid = item.get_local_linear_id();
    sycl::group_barrier(item.get_group());
    size_t address_space = 0;
    if (lid == 0) {
      address_space = out.prealloc(ref.load());
    }
    address_space = sycl::group_broadcast(item.get_group(), address_space, 0);
    for (size_t i = lid; i < ref; i += item.get_local_range(0)) {
      out.insert(context.data[i], address_space + i);
    }
  }

  sycl::local_accessor<type_t, 1> data;
  sycl::local_accessor<size_t, 1> tail;
  const size_t max_elems;
};

template <typename type_t>
class device_vector_frontier_t {
  friend class frontier_vector_t<type_t>;
public:
  device_vector_frontier_t(size_t max_size) : max_size(max_size) {}

  SYCL_EXTERNAL inline bool empty() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load() == 0;
  }

  SYCL_EXTERNAL inline size_t size() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.load();
  }

  SYCL_EXTERNAL inline bool insert(type_t val) const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    if (ref.load() >= max_size) {
      return false;
    }
    data[ref++] = val;
    return true;
  }
  
  SYCL_EXTERNAL inline bool insert(type_t val, size_t idx) const {
    if (idx >= *tail) {
      return false;
    }
    data[idx] = val;
    return true;
  }

  SYCL_EXTERNAL inline bool remove(type_t val) const {
    return false;
  }

  SYCL_EXTERNAL inline void clear() const {
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    ref = 0;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { // TODO: [!] check for max size
    sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(tail[0]);
    return ref.fetch_add(num_elems);
  }

protected:
  inline void set_tail(size_t new_tail) {
    *tail = new_tail;
  }
  inline size_t get_tail() const {
    return *tail;
  }

private:
  type_t* data;
  size_t* tail;
  size_t max_size;
};

template <typename type_t>
class frontier_vector_t {
public:
  using frontier_type = type_t;

  static void swap(frontier_vector_t<type_t>& a, frontier_vector_t<type_t>& b) {
    std::swap(a.vector.data, b.vector.data);
    std::swap(a.vector.tail, b.vector.tail);
  }

  // frontier_vector_t(sycl::queue& q, size_t num_elems) : q(q), vector(num_elems) {
  //   vector.data = sycl::malloc_shared<type_t>(num_elems, q);
  frontier_vector_t(sycl::queue& q, size_t num_elems) : q(q), vector(static_cast<size_t>(114613306 * 2)) {
    vector.data = sycl::malloc_shared<type_t>(static_cast<size_t>(114613306 * 2), q);
    vector.tail = sycl::malloc_shared<size_t>(1, q);
    vector.set_tail(0);
  }

  ~frontier_vector_t() {
    sycl::free(vector.data, q);
    sycl::free(vector.tail, q);
  }

  size_t get_num_active_elements() const {
    return vector.get_tail();
  }

  inline bool self_allocated() const { return true; }

  void get_active_elements(type_t*& elems) const {
    elems = vector.data;
  }

  /**
   * @brief Retrieves the active elements in the bitmap.
   * 
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
  */
  void get_active_elements(type_t*& elems, size_t& size) const {
    size = vector.get_tail();
    elems = vector.data;
  }

  inline bool empty() const {
    return vector.empty();
  }

  bool insert(type_t val) {
    vector.data[(*(vector.tail))++] = val;
    return true;
  }

  bool insert(type_t val, size_t idx) {
    if (idx >= vector.get_tail()) {
      return false;
    }
    vector.data[idx] = val;
    return true;
  }

  bool remove(type_t idx) {
    return false;
  }

  frontier_vector_t& operator=(const frontier_vector_t& other) {
    throw std::runtime_error("Not implemented");
  }

  inline void merge(frontier_bitmap_t<type_t>& other) {
    throw std::runtime_error("Not implemented");
  }

  inline void clear() {
    vector.set_tail(0);
  }

  const device_vector_frontier_t<type_t>& get_device_frontier() const {
    return vector;
  }

  const local_vector_context_t<type_t> get_local_context(sycl::handler& cgh) const {
    return std::move(local_vector_context_t<type_t>(types::detail::MAX_LOCAL_MEM_SIZE, cgh));
  }

private:
  sycl::queue& q;          ///< The SYCL queue used for memory allocation.
  device_vector_frontier_t<type_t> vector;
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph