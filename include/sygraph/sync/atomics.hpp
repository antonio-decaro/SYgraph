/*
 * Copyright (c) 2025 University of Salerno
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <sycl/sycl.hpp>

namespace sygraph {
namespace sync {

/**
 * @brief Performs an atomic fetch-and-add operation on the given pointer.
 *
 * This function uses SYCL's atomic_ref to perform an atomic fetch-and-add
 * operation on the value pointed to by the given pointer. The operation
 * is performed with relaxed memory order and device memory scope.
 *
 * @tparam T The type of the value to be added.
 * @param ptr A pointer to the value to be modified.
 * @param val The value to be added to the value pointed to by ptr.
 * @return The value of the pointed-to object immediately before the addition.
 */
template<typename T>
SYCL_EXTERNAL inline T atomicFetchAdd(T* ptr, T val) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.fetch_add(val);
}

/**
 * @brief Loads a value from the given pointer using SYCL atomic operations.
 *
 * This function uses SYCL's atomic_ref to load a value from the memory location
 * pointed to by the given pointer. The load operation is performed with relaxed
 * memory order and device memory scope.
 *
 * @tparam T The type of the value to be loaded.
 * @param ptr A pointer to the memory location from which the value is to be loaded.
 * @return The value loaded from the memory location.
 */
template<typename T>
SYCL_EXTERNAL inline T load(T* ptr) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.load();
}

/**
 * @brief Stores a value into the memory location pointed to by ptr using atomic operations.
 *
 * This function uses SYCL's atomic_ref to store the given value into the memory location
 * pointed to by ptr. The store operation is performed with relaxed memory order and device
 * memory scope.
 *
 * @tparam T The type of the value to be stored.
 * @param ptr A pointer to the memory location where the value will be stored.
 * @param val The value to be stored.
 */
template<typename T>
SYCL_EXTERNAL inline void store(T* ptr, T val) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  ref.store(val);
}

/**
 * @brief Computes the minimum of two values atomically.
 *
 * This function takes two pointers to values of type T and returns the minimum
 * of the two values using atomic operations. The atomic operations ensure that
 * the minimum computation is performed safely in a concurrent environment.
 *
 * @tparam T The type of the values.
 * @param v1 Pointer to the first value.
 * @param v2 Pointer to the second value.
 * @return The minimum of the two values.
 */
template<typename T>
SYCL_EXTERNAL inline T min(T* v1, T* v2) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref1(*v1);
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref2(*v2);
  return ref1.fetch_min(ref2.load());
}

/**
 * @brief Performs a compare-and-swap (CAS) operation on the given pointer.
 *
 * This function uses SYCL's atomic_ref to perform a CAS operation, which
 * compares the value at the given pointer with the expected value, and if
 * they are equal, replaces it with the desired value.
 *
 * @tparam T The type of the value pointed to by ptr.
 * @param ptr A pointer to the value to be compared and potentially swapped.
 * @param expected A reference to the value expected to be found at ptr.
 * @param desired The value to be set at ptr if the comparison is successful.
 * @return true if the swap was successful, false otherwise.
 */
/**
 * @brief Atomically sets `*ptr` to the maximum of its value and `val`.
 *
 * @return The value of `*ptr` before the operation.
 */
template<typename T>
SYCL_EXTERNAL inline T max(T* ptr, T val) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.fetch_max(val);
}

template<typename T>
SYCL_EXTERNAL inline bool cas(T* ptr, T& expected, T desired) {
  sycl::atomic_ref<T, sycl::memory_order::relaxed, sycl::memory_scope::device> ref(*ptr);
  return ref.compare_exchange_strong(expected, desired);
}
} // namespace sync
} // namespace sygraph