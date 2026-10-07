/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/utils/vector.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

namespace sygraph {
namespace frontier {
namespace detail {


template<typename T>
class FrontierBitmap;

template<typename T, typename B = types::bitmap_type_t>
class [[deprecated("BitmapDevice is deprecated, use MLB instead")]] BitmapDevice {
public:
  using bitmap_type = B;

  BitmapDevice(size_t num_elems) : _num_elems(num_elems) {
    _range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    _size = num_elems / _range + (num_elems % _range != 0);
  }

  /**
   * @brief Retrieves the size of the bitmap.
   *
   * @return The size of the bitmap.
   */
  SYCL_EXTERNAL inline size_t getBitmapSize() const { return _size; }

  /**
   * @brief Retrieves the number of elements in the bitmap.
   *
   * @return The number of elements in the bitmap.
   */
  SYCL_EXTERNAL inline size_t getNumElems() const { return _num_elems; }

  /**
   * @brief Retrieves the range of the bitmap.
   *
   * @return The range of the bitmap.
   */
  SYCL_EXTERNAL inline const size_t getBitmapRange() const { return _range; }

  /**
   * @brief Retrieves a pointer to the bitmap.
   *
   * @return A pointer to the bitmap.
   */
  SYCL_EXTERNAL bitmap_type* getData() const { return _data; }

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
  SYCL_EXTERNAL inline bool insert(T idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[getBitmapIndex(idx)]);
    ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % _range));
    return true;
  }

  SYCL_EXTERNAL inline size_t prealloc(size_t num_elems) const { return 0; }

  /**
   * @brief Sets the bit at the specified index to 0.
   *
   * @param idx The index of the bit to set.
   */
  SYCL_EXTERNAL inline bool remove(size_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[getBitmapIndex(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % _range)));
    return true;
  }

  /**
   * @brief Resets the bitmap by setting all bits to 0.
   * @note This function should be executed by a single work-item.
   */
  SYCL_EXTERNAL inline void reset() const {
    for (size_t i = 0; i < _size; i++) { _data[i] = static_cast<bitmap_type>(0); }
  }

  /**
   * @brief Resets the bit at the specified index to 0.
   * @detail All the bits from [bitmap[id], bitmap[id + 1]) are set to 0.
   *
   * @param id The index of the bitmap to reset.
   */
  SYCL_EXTERNAL inline void reset(size_t id) const { _data[id] = static_cast<bitmap_type>(0); }

  /**
   * @brief Checks if the bit at the specified index is set.
   *
   * @param idx The index of the bit to check.
   * @return True if the bit is set, false otherwise.
   */
  SYCL_EXTERNAL inline bool check(size_t idx) const { return _data[idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range)); }

  SYCL_EXTERNAL inline bool empty() const {
    bitmap_type any = static_cast<bitmap_type>(0);
    for (size_t i = 0; i < _size; i++) { any |= _data[i]; }
    return any == static_cast<bitmap_type>(0);
  }
  /**
   * @brief Retrieves the bitmap index for the specified index.
   *
   * @param idx The index.
   * @return The bitmap index.
   */
  SYCL_EXTERNAL inline const size_t getBitmapIndex(size_t idx) const { return idx / _range; }

  template<int Dim, typename group_t>
  SYCL_EXTERNAL size_t getNumActiveElements(sycl::nd_item<Dim> item, group_t group) const {
    size_t count = 0;
    auto id = item.get_local_linear_id();
    auto local_range = item.get_local_range(0);

    for (size_t i = id; i < _size; i += local_range) {
      for (bitmap_type j = 0; j < _range; j++) {
        if (_data[i] & (static_cast<bitmap_type>(1) << j)) { count++; }
      }
    }

    return sycl::reduce_over_group(group, count, sycl::plus<size_t>());
  }

  SYCL_EXTERNAL inline int* getOffsets() const { return _offsets; }

  SYCL_EXTERNAL inline uint32_t* getOffsetsSize() const { return _offsets_size; }

  friend class FrontierBitmap<T>;

protected:
  void setPtr(bitmap_type* ptr, int* offsets, uint32_t* offsets_size) {
    _data = ptr;
    this->_offsets = offsets;
    this->_offsets_size = offsets_size;
  }

  size_t _range;      ///< The range of the bitmap.
  size_t _num_elems;  ///< The number of elements in the bitmap.
  size_t _size;       ///< The size of the bitmap.
  bitmap_type* _data; ///< Pointer to the bitmap.

  int* _offsets;
  uint32_t* _offsets_size;
};

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
template<typename T>
class [[deprecated("BitmapDevice is deprecated, use MLB instead")]] FrontierBitmap {
public:
  /**
   * @brief Constructs a frontier_bitmap_t object.
   *
   * @param q The SYCL queue to use for memory allocation.
   * @param num_elems The number of elements in the bitmap.
   */
  FrontierBitmap(sycl::queue& q, size_t num_elems) : _queue(q), _bitmap(num_elems) { // TODO: [!] tune on bitmap size
    using bitmap_type = typename BitmapDevice<T>::bitmap_type;
    bitmap_type* ptr = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::shared>(_bitmap.getBitmapSize(), _queue);
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(_bitmap.getBitmapSize(), _queue);
    uint32_t* offsets_size = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::shared>(1, _queue);
    _queue.fill(ptr, static_cast<bitmap_type>(0), _bitmap.getBitmapSize()).wait();
    _queue.fill(offsets_size, static_cast<uint32_t>(0), 1).wait();
    _bitmap.setPtr(ptr, offsets, offsets_size);
  }

  using bitmap_type = typename BitmapDevice<T>::bitmap_type;

  FrontierBitmap(const FrontierBitmap&) = delete;
  FrontierBitmap& operator=(const FrontierBitmap&) = delete;

  FrontierBitmap(FrontierBitmap&& other) noexcept : _queue(other._queue), _bitmap(other._bitmap) { other._bitmap.setPtr(nullptr, nullptr, nullptr); }

  FrontierBitmap& operator=(FrontierBitmap&&) = delete;

  /**
   * @brief Destroys the frontier_bitmap_t object and frees the allocated memory.
   */
  ~FrontierBitmap() {
    auto* data = _bitmap.getData();
    memory::detail::releaseUSM(data, _queue);

    auto* offsets = _bitmap.getOffsets();
    memory::detail::releaseUSM(offsets, _queue);

    auto* offsets_size = _bitmap.getOffsetsSize();
    memory::detail::releaseUSM(offsets_size, _queue);

    _bitmap.setPtr(data, offsets, offsets_size);
  }

  size_t getBitmapSize() const { return _bitmap._size; }

  size_t getNumElems() const { return _bitmap._num_elems; }

  size_t getBitmapRange() const { return _bitmap._range; }

  bool selfAllocated() const { return false; }

  size_t getNumActiveElements() const { // TODO: [!!!] this kernel is too slow, we need a better way to count the number of active elements
    size_t* count = memory::detail::memoryAlloc<size_t, memory::space::shared>(1, _queue);

    sycl::nd_range<1> nd_range(128, 128); // TODO: [!] tune on these value

    _queue
        .submit([&](sycl::handler& h) {
          auto bitmap = this->getDeviceFrontier();

          h.parallel_for<class get_num_active_elements_kernel>(nd_range, [=](sycl::nd_item<1> item) {
            auto group = item.get_group();
            auto lcount = bitmap.getNumActiveElements(item, group);
            if (item.get_global_linear_id() == 0) { *count = lcount; }
          });
        })
        .wait();
    size_t ret = *count;
    sycl::free(count, _queue);
    return ret;
  }

  /**
   * @brief Retrieves the active elements in the bitmap.
   *
   * @param elems The array to store the active elements. It must be pre-allocated with shared-access.
   * @param active If true, it retrieves the active elements, otherwise the inactive elements.
   */
  void getActiveElements(T*& elems, size_t& size) const {
    constexpr size_t local = 32;
    sycl::range<1> local_size{local}; // TODO: [!] tuning on this value
    sycl::range<1> global_size{(_bitmap.getBitmapSize() > local ? _bitmap.getBitmapSize + local - (_bitmap.getBitmapSize % local) : local)};

    sycl::nd_range<1> nd_range(global_size, local_size);

    size_t bitmap_range = this->_bitmap.getBitmapRange();

    sycl::buffer<size_t, 1> g_tail_buffer(sycl::range<1>(1));

    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();

      sycl::local_accessor<T, 1> local_elems(bitmap_range * local_size, cgh);
      sycl::local_accessor<size_t, 1> l_tail(1, cgh);
      sycl::accessor tail_acc(g_tail_buffer, cgh, sycl::read_write);

      cgh.parallel_for<class get_active_elements_kernel>(
          nd_range, [=, bitmap_range = bitmap_range, bitmap_size = bitmap.size, data = bitmap.data](sycl::nd_item<1> item) {
            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> l_tail_ref(
                l_tail[0]); // TODO: check if acq_rel works
            sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> g_tail_ref(tail_acc[0]);

            auto lid = item.get_local_linear_id();
            auto gid = item.get_global_linear_id();
            auto global_size = item.get_global_range(0);
            auto group = item.get_group();
            auto group_id = item.get_group_linear_id();
            auto group_size = item.get_local_range(0);

            if (lid == 0) {
              l_tail_ref = 0;
              if (gid == 0) { g_tail_ref = 0; }
            }

            sycl::group_barrier(item.get_group());

            if (gid < bitmap_size) {
              auto elem = data[gid];

              for (T i = 0; i < bitmap_range; i++) {
                if (elem & (static_cast<bitmap_type>(1) << i)) { local_elems[l_tail_ref++] = i + gid * bitmap_range; }
              }
            }

            sycl::group_barrier(group);

            size_t our_slice = 0;
            if (lid == 0) { our_slice = g_tail_ref.fetch_add(l_tail_ref.load()); }
            our_slice = sycl::group_broadcast(group, our_slice, 0);
            for (size_t i = lid; i < l_tail_ref.load(); i += group_size) { elems[our_slice + i] = local_elems[i]; }
          });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "getActiveElements");
#endif
    size = g_tail_buffer.get_host_access()[0];
  }

  bool empty() const { return _bitmap.empty(); }

  bool check(size_t idx) const { return _bitmap.check(idx); }

  bool insert(size_t idx) {
    _queue
        .submit([&](sycl::handler& cgh) {
          auto bitmap = this->getDeviceFrontier();
          cgh.single_task([=]() { bitmap.insert(idx); });
        })
        .wait();
    return true;
  }

  bool remove(size_t idx) {
    _queue
        .submit([&](sycl::handler& cgh) {
          auto bitmap = this->getDeviceFrontier();
          cgh.single_task([=]() { bitmap.remove(idx); });
        })
        .wait();
    return true;
  }

  /**
   * Merges the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to merge with.
   * @return The event associated with the operation.
   * @post The current frontier contains the union of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::Event merge(const FrontierBitmap<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class merge_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                           [=](sycl::id<1> idx) { bitmap.getData()[idx] |= other_bitmap.getData()[idx]; });
    });
  }

  /**
   * @brief Intersects the contents of the current bitmap frontier with the specified frontier.
   *
   * @param other The frontier to intersect with.
   * @return The event associated with the operation.
   * @post The current frontier contains the intersection of the current frontier and the specified frontier. The specified frontier is not modified.
   */
  sygraph::Event intersect(const FrontierBitmap<T>& other) {
    return _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<class intersect_bitmap_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                               [=](sycl::id<1> idx) { bitmap.getData()[idx] &= other_bitmap.getData()[idx]; });
    });
  }

  /**
   * @brief Clears the bitmap by setting all bits to 0. It performs only a single memory operation.
   * @note This function should be called only on the host-side.
   */
  void clear() {
    _queue.fill(_bitmap._data, static_cast<bitmap_type>(0), _bitmap._size).wait();
    _queue.fill(_bitmap._offsets_size, 0, 1).wait();
  }

  const BitmapDevice<T, bitmap_type>& getDeviceFrontier() const { return _bitmap; }

  size_t computeActiveFrontier() const {
    sycl::range<1> local_range{types::detail::COMPUTE_UNIT_SIZE};
    size_t size = _bitmap.getBitmapSize();
    sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};

    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();

      sycl::local_accessor<int, 1> local_offsets(local_range[0], cgh);
      sycl::local_accessor<uint32_t, 1> local_size(1, cgh);
      bitmap._offsets_size[0] = 0;

      cgh.parallel_for(sycl::nd_range<1>{global_range, local_range},
                       [=, offsets_size = bitmap._offsets_size, offsets = bitmap._offsets](sycl::nd_item<1> item) {
                         int gid = item.get_global_linear_id();
                         size_t lid = item.get_local_linear_id();
                         auto group = item.get_group();

                         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> local_size_ref(local_size[0]);
                         sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

                         if (group.leader()) { local_size_ref.store(0); }
                         sycl::group_barrier(group);

                         if (gid < size && bitmap.getData()[gid] != 0) { local_offsets[local_size_ref++] = gid; }
                         sycl::group_barrier(group);

                         size_t data_offset = 0;
                         if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
                         data_offset = sycl::group_broadcast(group, data_offset, 0);
                         if (lid < local_size_ref.load()) { offsets[data_offset + lid] = local_offsets[lid]; }
                       });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeActiveFrontier");
#endif
    return _bitmap._offsets_size[0];
  }

  static void swap(FrontierBitmap<T>& a, FrontierBitmap<T>& b) { std::swap(a._bitmap, b._bitmap); }

private:
  sycl::queue& _queue;                  ///< The SYCL queue used for memory allocation.
  BitmapDevice<T, bitmap_type> _bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace sygraph
