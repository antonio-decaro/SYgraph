/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sygraph/operators/config.hpp>
#include <sygraph/sycl/event.hpp>
#include <sygraph/utils/device.hpp>
#include <sygraph/utils/memory.hpp>
#include <sygraph/utils/types.hpp>
#include <sygraph/utils/vector.hpp>
#ifdef ENABLE_PROFILING
#include <sygraph/utils/profiler.hpp>
#endif

namespace sygraph {
namespace frontier {

namespace detail {

class mlb_compute_active_frontier_kernel;
class is_mlb_frontier_empty_kernel;
class compute_size_mlb_frontier_kernel;
class merge_mlb_frontier_kernel;
class intersect_mlb_frontier_kernel;


template<size_t Levels, typename B = types::bitmap_type_t>
class BitmapState {
public:
  std::array<size_t, Levels> size;
  std::array<std::vector<B>, Levels> data;
};


template<typename DeviceFrontier>
concept DeviceFrontierConcept = requires(DeviceFrontier f) {
  { f.getBitmapSize() } -> std::convertible_to<size_t>;
  { f.getNumElems() } -> std::convertible_to<size_t>;
  { f.getBitmapRange() } -> std::convertible_to<size_t>;
  { f.getData() } -> std::convertible_to<typename DeviceFrontier::bitmap_type*>;
  { f.set(0, true) } -> std::convertible_to<void>;
  { f.insert(0) } -> std::convertible_to<bool>;
  { f.remove(0) } -> std::convertible_to<bool>;
  { f.reset() } -> std::convertible_to<void>;
  { f.reset(0) } -> std::convertible_to<void>;
  { f.check(0) } -> std::convertible_to<bool>;
  { f.empty() } -> std::convertible_to<bool>;
  { f.getBitmapIndex(0) } -> std::convertible_to<size_t>;
  { f.getOffsets() } -> std::convertible_to<int*>;
  { f.getOffsetsSize() } -> std::convertible_to<uint32_t*>;
};

template<typename T, size_t Levels, DeviceFrontierConcept DeviceFrontier>
class FrontierMLB;

template<typename T, size_t Levels, typename B = types::bitmap_type_t>
class MLBDevice {
public:
  using bitmap_type = B;

  MLBDevice(size_t num_elems) : _num_elems(num_elems) {
    _range = sizeof(bitmap_type) * sygraph::types::detail::byte_size;
    _size[0] = (num_elems / _range) + static_cast<size_t>(num_elems % _range != 0);

    for (uint16_t i = 1; i < Levels; i++) { _size[i] = (_size[i - 1] / _range) + (_size[i - 1] % _range != 0); }
  }

  SYCL_EXTERNAL inline uint32_t getBitmapSize() const { return _size[0]; }

  SYCL_EXTERNAL inline uint32_t getNumElems() const { return _num_elems; }

  SYCL_EXTERNAL inline const uint32_t getBitmapRange() const { return _range; }

  SYCL_EXTERNAL inline bitmap_type* getData() const { return _data[0]; }

  SYCL_EXTERNAL inline void set(uint32_t idx, bool val) const {
    if (val) {
      insert(idx);
    } else {
      remove(idx);
    }
  }

  SYCL_EXTERNAL inline bool insert(T idx) const {
#pragma unroll
    for (uint16_t i = 0; i < Levels; i++) {
      T lidx = idx;
      for (uint16_t _ = 0; _ < i; _++) { lidx /= _range; } // the index must be divided by the range^level
      if (!(_data[i][getBitmapIndex(lidx)] & (static_cast<bitmap_type>(1) << (lidx % _range)))) {
        sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[i][getBitmapIndex(lidx)]);
        ref |= static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (lidx % _range));
      }
    }
    return true;
  }

  SYCL_EXTERNAL inline bool remove(uint32_t idx) const {
    sycl::atomic_ref<bitmap_type, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(_data[0][getBitmapIndex(idx)]);
    ref &= ~(static_cast<bitmap_type>(static_cast<bitmap_type>(1) << (idx % _range)));
    return true;
  }

  SYCL_EXTERNAL inline void reset() const {
    for (uint16_t i = 0; i < _size; i++) { _data[i] = static_cast<bitmap_type>(0); }
  }

  SYCL_EXTERNAL inline void reset(uint32_t id) const { _data[id] = static_cast<bitmap_type>(0); }

  SYCL_EXTERNAL inline bool check(uint32_t idx) const { return _data[0][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range)); }

  SYCL_EXTERNAL inline bool empty() const {
    bitmap_type count = static_cast<bitmap_type>(0);
    for (auto i = 0; i < _size[Levels - 1]; i++) { count += _data[Levels - 1][i]; }
    return count == static_cast<bitmap_type>(0);
  }

  SYCL_EXTERNAL inline bool empty(uint32_t el_idx, uint16_t level) const { return _data[level][el_idx]; }

  SYCL_EXTERNAL inline const uint32_t getBitmapIndex(uint32_t idx) const { return idx / _range; }

  SYCL_EXTERNAL inline int* getOffsets() const { return _offsets; }

  SYCL_EXTERNAL inline uint32_t* getOffsetsSize() const { return _offsets_size; }

  SYCL_EXTERNAL inline uint32_t getBitmapSize(const uint level) const { return _size[level]; }

  SYCL_EXTERNAL inline bitmap_type* getData(const uint level) const { return _data[level]; }

  SYCL_EXTERNAL inline bool check(const uint level, uint32_t idx) const {
    return _data[level][idx / _range] & (static_cast<bitmap_type>(1) << (idx % _range));
  }

  void setData(bitmap_type* data[Levels]) {
    for (uint16_t i = 0; i < Levels; i++) { this->_data[i] = data[i]; }
  }

  void setData(const uint level, bitmap_type* data) { this->_data[level] = data; }

  void setOffsets(int* offsets) { this->_offsets = offsets; }

  void setOffsetsSize(uint32_t* offsets_size) { this->_offsets_size = offsets_size; }

protected:
  uint _range;                ///< The range of the bitmap.
  uint32_t _num_elems;        ///< The number of elements in the bitmap.
  uint32_t _size[Levels];     ///< The size of the bitmap.
  bitmap_type* _data[Levels]; ///< Pointer to the bitmap.

  int* _offsets;
  uint32_t* _offsets_size;
};

template<typename T, size_t Levels = 2, DeviceFrontierConcept DeviceFrontier = MLBDevice<T, Levels>>
class FrontierMLB {
public:
  using bitmap_type = typename DeviceFrontier::bitmap_type;
  using device_frontier_type = DeviceFrontier;
  using frontier_state_type = BitmapState<Levels, bitmap_type>;

  FrontierMLB(sycl::queue& q, size_t num_elems) : _queue(q), _bitmap(num_elems) {
    bitmap_type* ptr[Levels];
#pragma unroll
    for (size_t i = 0; i < Levels; i++) {
      size_t size = _bitmap.getBitmapSize(i);
      ptr[i] = sygraph::memory::detail::memoryAlloc<bitmap_type, memory::space::device>(size, _queue);
      _queue.fill(ptr[i], static_cast<bitmap_type>(0), size);
    }
    _queue.wait();
    int* offsets = sygraph::memory::detail::memoryAlloc<int, memory::space::device>(_bitmap.getBitmapSize(), _queue);
    uint32_t* offsets_size = sygraph::memory::detail::memoryAlloc<uint32_t, memory::space::device>(1, _queue);
    auto size = _bitmap.getBitmapSize();
    _queue.fill(offsets_size, static_cast<uint32_t>(0), 1).wait();

    _bitmap.setData(ptr);
    _bitmap.setOffsets(offsets);
    _bitmap.setOffsetsSize(offsets_size);
  }

  FrontierMLB(const FrontierMLB&) = delete;
  FrontierMLB& operator=(const FrontierMLB&) = delete;

  FrontierMLB(FrontierMLB&& other) noexcept : _queue(other._queue), _bitmap(other._bitmap) {
    other._bitmap = DeviceFrontier(other._bitmap.getNumElems());
    for (size_t i = 0; i < Levels; i++) { other._bitmap.setData(i, nullptr); }
    other._bitmap.setOffsets(nullptr);
    other._bitmap.setOffsetsSize(nullptr);
  }

  FrontierMLB& operator=(FrontierMLB&&) = delete;

  ~FrontierMLB() {
    for (size_t i = 0; i < Levels; i++) {
      auto* data = _bitmap.getData(i);
      memory::detail::releaseUSM(data, _queue);
      _bitmap.setData(i, data);
    }
    auto* offsets = _bitmap.getOffsets();
    memory::detail::releaseUSM(offsets, _queue);
    _bitmap.setOffsets(offsets);

    auto* offsets_size = _bitmap.getOffsetsSize();
    memory::detail::releaseUSM(offsets_size, _queue);
    _bitmap.setOffsetsSize(offsets_size);
  }

  size_t getBitmapSize() const { return _bitmap.getBitmapSize(); }

  size_t getNumElems() const { return _bitmap.getNumElems(); }

  size_t getBitmapRange() const { return _bitmap.getBitmapRange(); }

  bool selfAllocated() const { return false; }

  bool empty() const {
    auto bitmap = this->getDeviceFrontier();

    const size_t local_size = types::detail::COMPUTE_UNIT_SIZE;
    const size_t global_size = local_size * sygraph::detail::device::getNumComputeUnits(_queue);

    size_t bitmap_size = bitmap.getBitmapSize(Levels - 1);
    size_t moduled_size = bitmap_size % local_size ? bitmap_size + local_size - (bitmap_size % local_size) : bitmap_size;

    bool check = false;
    sycl::buffer<bool, 1> check_buf(&check, sycl::range<1>(1));
    check_buf.set_write_back(false);

    auto e = _queue.submit([&](sycl::handler& cgh) {
      sycl::accessor check_acc(check_buf, cgh, sycl::read_write);

      cgh.parallel_for<is_mlb_frontier_empty_kernel>(sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
        sycl::group<1> group = item.get_group();
        bool tmp = false;
        for (auto i = item.get_global_linear_id(); i < moduled_size && !tmp && !check_acc[0]; i += global_size) {
          tmp = sycl::any_of_group(group, i < bitmap_size ? bitmap.getData(Levels - 1)[i] : 0, [](bitmap_type val) { return val != 0; });
        }
        if (group.leader() && !check_acc[0] && tmp) check_acc[0] = true;
      });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "isFrontierEmpty");
#endif
    return !check_buf.get_host_access()[0];
  }

  bool check(size_t idx) const {
    sycl::buffer<bool, 1> check_buf(sycl::range<1>(1));
    auto e = _queue.submit([&](sycl::handler& cgh) {
      sycl::accessor check_acc(check_buf, cgh, sycl::write_only);
      auto bitmap = this->getDeviceFrontier();
      cgh.single_task([=]() { check_acc[0] = bitmap.check(idx); });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "checkFrontierElement");
#endif
    sycl::host_accessor check_acc(check_buf);
    return check_acc[0];
  }

  bool insert(size_t idx) {
    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      cgh.single_task([=]() { bitmap.insert(idx); });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "insertFrontierElement");
#endif
    return true;
  }

  bool remove(size_t idx) {
    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      cgh.single_task([=]() { bitmap.remove(idx); });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "removeFrontierElement");
#endif
    return true;
  }

  inline const size_t size() const {
    size_t active_count = 0;
    sycl::buffer<size_t, 1> size_buf(&active_count, sycl::range<1>(1));
    size_t frontier_size = this->getBitmapSize();
    size_t bitmap_range = this->getBitmapRange();
    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto size_acc = size_buf.get_access<sycl::access::mode::read_write>(cgh);
      cgh.parallel_for<compute_size_mlb_frontier_kernel>(sycl::range<1>{frontier_size}, [=](sycl::id<1> idx) {
        sycl::atomic_ref<size_t, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(size_acc[0]);
        size_t num_active_nodes = 0;
        bitmap_type t = bitmap.getData()[idx];
        for (size_t i = 0; i < bitmap_range; i++) {
          if (t & (static_cast<bitmap_type>(1) << i)) { num_active_nodes++; }
        }
        ref += num_active_nodes;
      });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeFrontierSize");
#endif
    sycl::host_accessor<size_t, 1> size_acc(size_buf);
    return size_acc[0];
  }

  void merge(const FrontierMLB<T>& other) {
    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<merge_mlb_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                  [=](sycl::id<1> idx) { bitmap.getData()[idx] |= other_bitmap.getData()[idx]; });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "mergeFrontier");
#endif
  }

  void intersect(const FrontierMLB<T>& other) {
    auto e = _queue.submit([&](sycl::handler& cgh) {
      auto bitmap = this->getDeviceFrontier();
      auto other_bitmap = other.getDeviceFrontier();
      cgh.parallel_for<intersect_mlb_frontier_kernel>(sycl::range<1>(bitmap.getBitmapSize()),
                                                      [=](sycl::id<1> idx) { bitmap.getData()[idx] &= other_bitmap.getData()[idx]; });
    });
    e.wait();
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "intersectFrontier");
#endif
  }

  sygraph::frontier::detail::BitmapState<Levels, bitmap_type> saveState() {
    sygraph::frontier::detail::BitmapState<Levels, bitmap_type> state;
#pragma unroll
    for (size_t i = 0; i < Levels; i++) {
      state.size[i] = _bitmap.getBitmapSize(i);
      state.data[i].resize(state.size[i]);
      auto e = _queue.copy(_bitmap.getData(i), state.data[i].data(), state.size[i]);
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "saveFrontierState_level_" + std::to_string(i));
#endif
    }

    _queue.wait();
    return state;
  }

  void loadState(const sygraph::frontier::detail::BitmapState<Levels, bitmap_type>& state) {
#pragma unroll
    for (size_t i = 0; i < Levels; i++) {
      assert(state.size[i] == _bitmap.getBitmapSize(i));
      auto e = _queue.copy(state.data[i].data(), _bitmap.getData(i), state.size[i]);
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "loadFrontierState_level_" + std::to_string(i));
#endif
    }
    _queue.wait();
  }

  void clear() {
    sycl::event e;
    for (size_t i = 0; i < Levels; i++) {
      e = _queue.fill(_bitmap.getData(i), static_cast<bitmap_type>(0), _bitmap.getBitmapSize(i));
#ifdef ENABLE_PROFILING
      sygraph::Profiler::addEvent(e, "clearFrontierLevel_" + std::to_string(i));
#endif
    }
    e = _queue.fill(_bitmap.getOffsetsSize(), static_cast<uint32_t>(0), 1);
#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "clearFrontierOffsets");
#endif
    _queue.wait();
  }

  const DeviceFrontier& getDeviceFrontier() const { return _bitmap; }

  /**
   * @brief Computes the active frontier by populating the offsets array with the indices of active elements.
   * @param invert If true, computes the inactive frontier instead (for pull-based advance operations).
   */
  sycl::event computeActiveFrontier(bool invert = false) const {
    sycl::range<1> local_range{types::detail::COMPUTE_UNIT_SIZE};
    auto bitmap = this->getDeviceFrontier();
    size_t size = bitmap.getBitmapSize(1);
    size_t level0_size = bitmap.getBitmapSize(0);
    uint32_t range = bitmap.getBitmapRange();
    // sycl::range<1> global_range{(size > local_range[0] ? size + local_range[0] - (size % local_range[0]) : local_range[0])};
    size_t global_size = sygraph::detail::device::getNumComputeUnits(_queue) * local_range[0];
    sycl::range<1> global_range{global_size};

    auto e = this->_queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<int, 1> local_offsets(local_range[0] * range, cgh);
      sycl::local_accessor<uint32_t, 1> local_size(1, cgh);

      cgh.parallel_for<mlb_compute_active_frontier_kernel>(
          sycl::nd_range<1>{global_range, local_range},
          [=, offsets_size = bitmap.getOffsetsSize(), offsets = bitmap.getOffsets()](sycl::nd_item<1> item) {
            // if (offsets_size[0] > 0) { return; } // TODO optimize for multiple calls on the same frontier
            int gid = item.get_global_linear_id();
            auto group = item.get_group();
            if (item.get_global_linear_id() == 0) { offsets_size[0] = 0; }
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::work_group> local_size_ref(local_size[0]);
            sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device> offsets_size_ref{offsets_size[0]};

            if (group.leader()) { local_size_ref.store(0); }
            sycl::group_barrier(group);
            for (uint32_t gid = item.get_global_linear_id(); gid < size; gid += item.get_global_range(0)) {
              bitmap_type data = bitmap.getData(1)[gid];
              for (size_t i = 0; i < range && i + (gid * range) < level0_size; i++) {
                bool is_active = (data & (static_cast<bitmap_type>(1) << i)) != 0;
                uint32_t pos;
                if ((!invert && !is_active)
                    || (invert && is_active && (bitmap.getData(0)[i + (gid * range)] == std::numeric_limits<bitmap_type>::max()))) {
                  continue;
                }

                local_offsets[local_size_ref++] = static_cast<int>(i + (gid * range));
              }
            }

            sycl::group_barrier(group);

            size_t data_offset = 0;
            if (group.leader()) { data_offset = offsets_size_ref.fetch_add(local_size_ref.load()); }
            data_offset = sycl::group_broadcast(group, data_offset, 0);
            for (size_t i = item.get_local_linear_id(); i < local_size_ref.load(); i += item.get_local_range(0)) {
              offsets[data_offset + i] = local_offsets[i];
            }
          });
    });

#ifdef ENABLE_PROFILING
    sygraph::Profiler::addEvent(e, "computeActiveFrontier");
#endif
    return e;
  }

  static void swap(FrontierMLB<T>& a, FrontierMLB<T>& b) { std::swap(a._bitmap, b._bitmap); }

protected:
  sycl::queue& _queue;    ///< The SYCL queue used for memory allocation.
  DeviceFrontier _bitmap; ///< The bitmap.
};


} // namespace detail
} // namespace frontier
} // namespace sygraph
