#pragma once

#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/vector.hpp>
#include <sygraph/utils/types.hpp>

namespace sygraph {
inline namespace v0 {
namespace frontier {
namespace detail {


template <typename type_t>
class frontier_bitmap_t;


template <typename type_t, typename bitmap_t = uint64_t>
class bitmap_device_t {
public:
  using bitmap_type = bitmap_t;

  bitmap_device_t(size_t num_elems) : num_elems(num_elems)  {
    range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    size = num_elems / range + (num_elems % range != 0);
  }
  
  /**
   * @brief Retrieves the size of the bitmap.
   * 
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t get_bitmap_size() const {
    return size;
  }

  /**
   * @brief Retrieves the number of elements in the bitmap.
   * 
   * @return The number of elements in the bitmap.
   */
  SYCL_EXTERNAL inline size_t get_num_elems() const {
    return num_elems;
  }

  /**
   * @brief Retrieves the range of the bitmap.
   * 
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t get_bitmap_range() const {
    return range;
  }

  /**
   * @brief Retrieves a pointer to the bitmap.
   * 
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* get_data() const {
    return data;
  }

  /**
   * @brief Sets the bit at the specified index to the specified value.
   * 
   * @param idx The index of the bit to set.
   * @param val The value to set the bit to.
   */
  SYCL_EXTERNAL inline void set(size_t idx, bool val) const {
    if (val) {
      insert(idx);
    } else {
      remove(idx);
    }
  }

  /**
   * @brief Sets the bit at the specified index to 1.
   * 
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline void insert(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref(data[get_bitmap_index(idx)]);
    ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % range));
  }

  /**
   * @brief Sets the bit at the specified index to 0.
   * 
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline void remove(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::work_group> ref(data[get_bitmap_index(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % range)));
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   * @note This function should be executed by a single work-item.
   */
  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < size; i++) {
      data[i] = static_cast<bitmap_type>(0);
    }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * @details All the bits from [bitmap[id], bitmap[id + 1]) are set to 0.
   * 
   * @param id The index of the bitmap to reset.
   */
  SYCL_EXTERNAL inline void reset(size_t id) const {
    data[id] = static_cast<bitmap_type>(0);
  }

  /**
   * @brief Checks if the bit at the specified index is set.
   * 
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  SYCL_EXTERNAL inline bool check(size_t idx) const {

    return data[idx / range] & (static_cast<bitmap_type>(1) << (idx % range));
  }

  SYCL_EXTERNAL inline bool empty() const {
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < size; i++) {
      count += data[i];
    }
    return count == static_cast<bitmap_type>(0);
  }

  /**
   * @brief Retrieves the bitmap index for the specified index.
   * 
   * @param idx The index.
   * @return The bitmap index.
   */
  SYCL_EXTERNAL inline const size_t get_bitmap_index(size_t idx) const {
    return idx / range;
  }

  template<int Dim, typename group_t>
  SYCL_EXTERNAL size_t get_num_active_elements(sycl::nd_item<Dim> item, group_t group) const {
    size_t count = 0;
    auto id = item.get_local_linear_id();
    auto local_range = item.get_local_range(0);

    for (size_t i = id; i < size; i += local_range) {
      for (bitmap_type j = 0; j < range; j++) {
        if (data[i] & (static_cast<bitmap_type>(1) << j)) {
          count++;
        }
      }
    }

    return sycl::reduce_over_group(group, count, sycl::plus<size_t>());
  }

  friend class frontier_bitmap_t<type_t>;
private:
  void set_ptr(bitmap_type* ptr) {
    data = ptr;
  }

  size_t range;            ///< The range of the bitmap.
  size_t num_elems;        ///< The number of elements in the bitmap.
  size_t size;             ///< The size of the bitmap.
  bitmap_type* data;     ///< Pointer to the bitmap.
};

template <typename type_t>
/**
 * @class frontier_bitmap_t
 * @brief Represents a bitmap frontier used in SYgraph.
 * 
 * The frontier_bitmap_t class provides functionality to manage a bitmap frontier in SYgraph.
 * It allows allocating and freeing memory for the bitmap, as well as retrieving the bitmap range.
 */
/**
 * @class frontier_bitmap_t
 * @brief Represents a bitmap frontier used in SYgraph.
 * 
 * The frontier_bitmap_t class provides functionality for managing a bitmap frontier,
 * which is used to track the active elements in a graph traversal algorithm.
 * 
 * @tparam bitmap_type The type of the bitmap.
 */
class frontier_bitmap_t {
public:
  /**
   * @brief Constructs a frontier_bitmap_t object.
   * 
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  // TODO: tune on bitmap size
  frontier_bitmap_t(sycl::queue& q, size_t num_elems) : q(q), bitmap(num_elems) {
    using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;
    bitmap_type* ptr = sygraph::memory::detail::memory_alloc<bitmap_type, memory::space::shared>(bitmap.get_bitmap_size(), q);
    auto size = bitmap.get_bitmap_size();
    q.memset(ptr, static_cast<bitmap_type>(0), size).wait();
    bitmap.set_ptr(ptr);
  }

  using bitmap_type = typename bitmap_device_t<type_t>::bitmap_type;

  /**
   * @brief Destroys the frontier_bitmap_t object and frees the allocated memory.
   */
  ~frontier_bitmap_t() {
    sycl::free(bitmap.get_data(), q);
  }

  inline size_t get_bitmap_size() const {
    return bitmap.size;
  }

  inline size_t get_num_elems() const {
    return bitmap.num_elems;
  }

  inline size_t get_bitmap_range() const {
    return bitmap.range;
  }

  size_t get_num_active_elements() const {
    size_t* count = memory::detail::memory_alloc<size_t, memory::space::shared>(1, q);

    sycl::nd_range<1> nd_range(32, 32); // TODO: tune on these value

    q.submit([&](sycl::handler& h) {
      auto bitmap = this->get_device_frontier();
      sycl::stream out(1024, 256, h);
      h.parallel_for(nd_range, [=](sycl::nd_item<1> item) {
        auto group = item.get_group();
        auto lcount = bitmap.get_num_active_elements(item, group);
        if (item.get_global_linear_id() == 0) {
          *count = lcount;
        }
      });
    }).wait();
    size_t ret = *count;
    sycl::free(count, q);
    return ret;
  }

  /**
   * @brief Retrieves the active elements in the bitmap.
   * 
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
  */
  void get_active_elements(type_t* elems) const {
    constexpr size_t local = 32;
    sycl::range<1> local_size {local}; // TODO: tuning on this value
    sycl::range<1> global_size {(bitmap.size > local ? bitmap.size + local - (bitmap.size % local) : local)};

    sycl::nd_range<1> nd_range(global_size, local_size);

    size_t bitmap_range = this->bitmap.get_bitmap_range();

    sycl::buffer<type_t, 1> active_elems(sycl::range<1>(1));

    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();

      sycl::local_accessor<type_t, 1> local_elems(bitmap_range * local_size, cgh);
      sycl::local_accessor<size_t, 1> local_count(1, cgh);
      sycl::accessor global_count(active_elems, cgh, sycl::read_write);

      cgh.parallel_for(nd_range, [=, bitmap_range=bitmap_range, bitmap_size=bitmap.size, data=bitmap.data](sycl::nd_item<1> item) {
        sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> lcount(local_count[0]); // TODO: check if it works
        sycl::atomic_ref<type_t, sycl::memory_order::relaxed, sycl::memory_scope::device> gcount(global_count[0]); // TODO: check if it works
        if (item.get_global_id(0) == 0) {
          gcount = 0;
        }
        auto lid = item.get_local_linear_id();
        auto gid = item.get_global_linear_id();
        auto global_size = item.get_global_range(0);
        auto group = item.get_group();
        auto group_id = item.get_group_linear_id();
        auto group_size = item.get_local_range(0);

        if (gid < bitmap_size) {
          auto elem = data[gid];

          for (type_t i = 0; i < bitmap_range; i++) {
            if (elem & (static_cast<bitmap_type>(1) << i)) {
              local_elems[lcount++] = i + gid * bitmap_range;
            }
          }
        }

        sycl::group_barrier(group);

        if (lid == 0) { // TODO: It might be done in a more efficient way. In this way there is a lot of pressure on the gcount and elems array.
          for (size_t i = 0; i < lcount; i++) {
            elems[gcount++] = local_elems[i];
          }
          lcount = 0;
        }
      });
    }).wait();
  }

  inline bool empty() const {
    return bitmap.empty();
  }

  bool check(size_t idx) const {
    return bitmap.check(idx);
  }

  void insert(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.insert(idx);
      });
    }).wait();
  }

  void remove(size_t idx) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      cgh.single_task([=]() {
        bitmap.remove(idx);
      });
    }).wait();
  }

  //operator =
  frontier_bitmap_t& operator=(const frontier_bitmap_t& other) {
    if (this == &other) {
      return *this;
    }
    q.copy(other.bitmap.data, this->bitmap.data, bitmap.size).wait();
    return *this;
  }

  /**
   * Merges the contents of the current bitmap frontier with the specified frontier.
   * 
   * @param other The frontier to merge with.
   * @post The current frontier contains the union of the current frontier and the specified frontier. The specified frontier is not modified.
  */
  inline void merge(frontier_bitmap_t<type_t>& other) {
    q.submit([&](sycl::handler& cgh) {
      auto bitmap = this->get_device_frontier();
      auto other_bitmap = other.get_device_frontier();
      cgh.parallel_for(sycl::range<1>(bitmap.size), [=](sycl::id<1> idx) {
        bitmap.data[idx] |= other_bitmap.data[idx];
      });
    }).wait();
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  inline void clear() {
    q.fill(bitmap.data, static_cast<bitmap_type>(0), bitmap.size).wait();
  }

  const bitmap_device_t<type_t, bitmap_type>& get_device_frontier() const {
    return bitmap;
  }

private:
  sycl::queue& q;          ///< The SYCL queue used for memory allocation.
  bitmap_device_t<type_t, bitmap_type> bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace v0
} // namespace sygraph