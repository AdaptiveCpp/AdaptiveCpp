/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_BUFFER_EXPLICIT_BEHAVIOR_HPP
#define HIPSYCL_BUFFER_EXPLICIT_BEHAVIOR_HPP


#include "buffer.hpp"
#include "hipSYCL/sycl/info/queue.hpp"
#include "queue.hpp"

namespace hipsycl {
namespace sycl {

/// Only internal storage, no writeback, blocking destructor
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_buffer(sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      r, property_list{detail::buffer_policy::use_external_storage{false},
                       detail::buffer_policy::writes_back{false},
                       detail::buffer_policy::destructor_waits{true}}};
}

template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_buffer(const T* ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      ptr, r,
      property_list{detail::buffer_policy::use_external_storage{false},
                    detail::buffer_policy::writes_back{false},
                    detail::buffer_policy::destructor_waits{true}}};
}

/// Only internal storage, no writeback, non-blocking destructor
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_buffer(sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      r, property_list{detail::buffer_policy::use_external_storage{false},
                       detail::buffer_policy::writes_back{false},
                       detail::buffer_policy::destructor_waits{false}}};
}

template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_buffer(const T* ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      ptr, r,
      property_list{detail::buffer_policy::use_external_storage{false},
                    detail::buffer_policy::writes_back{false},
                    detail::buffer_policy::destructor_waits{false}}};
}

/// External storage, writes back, blocking destructor
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_writeback_view(T* host_view_ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      host_view_ptr,
      r, property_list{detail::buffer_policy::use_external_storage{true},
                       detail::buffer_policy::writes_back{true},
                       detail::buffer_policy::destructor_waits{true}}};
}

/// External storage, writes back, non-blocking destructor
/// The queue can be used to wait for the writeback to complete.
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_writeback_view(T* host_view_ptr, sycl::range<Dim> r, const sycl::queue& q) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      host_view_ptr,
      r, property_list{detail::buffer_policy::use_external_storage{true},
                       detail::buffer_policy::writes_back{true},
                       detail::buffer_policy::destructor_waits{false},
                       property::buffer::AdaptiveCpp_write_back_node_group{
                         q.get_info<info::queue::AdaptiveCpp_node_group>()
                       }}};
}

template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_writeback_view(T* host_view_ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      host_view_ptr,
      r, property_list{detail::buffer_policy::use_external_storage{true},
                       detail::buffer_policy::writes_back{true},
                       detail::buffer_policy::destructor_waits{false}}};
}

/// External stroage, does not write back, blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_view(T* host_view_ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      host_view_ptr,
      r, property_list{detail::buffer_policy::use_external_storage{true},
                       detail::buffer_policy::writes_back{false},
                       detail::buffer_policy::destructor_waits{true}}};
}

/// External stroage, does not write back, non-blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_view(T* host_view_ptr, sycl::range<Dim> r) {
  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      host_view_ptr,
      r, property_list{detail::buffer_policy::use_external_storage{true},
                       detail::buffer_policy::writes_back{false},
                       detail::buffer_policy::destructor_waits{false}}};
}

/// USM interop

/// External storage, does not write back, blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_view(
    const std::vector<buffer_allocation::tracked_descriptor<T>>
        &input_allocations,
    sycl::range<Dim> r) {

  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      input_allocations, r,
      property_list{detail::buffer_policy::use_external_storage{true},
                    detail::buffer_policy::writes_back{false},
                    detail::buffer_policy::destructor_waits{true}}};
}

/// External storage, does not write back, non-blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_view(
    const std::vector<buffer_allocation::tracked_descriptor<T>>
        &input_allocations,
    sycl::range<Dim> r) {

  return buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>{
      input_allocations, r,
      property_list{detail::buffer_policy::use_external_storage{true},
                    detail::buffer_policy::writes_back{false},
                    detail::buffer_policy::destructor_waits{false}}};
}

/// External storage, does not write back, blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_sync_usm_view(const std::vector<buffer_allocation::tracked_descriptor<T>>
                       &input_allocations,
                   sycl::range<Dim> r) {
  return make_sync_view(input_allocations, r);
}

/// External storage, does not write back, non-blocking
template <class T, int Dim>
buffer<T, Dim, buffer_allocator<std::remove_const_t<T>>>
make_async_usm_view(const std::vector<buffer_allocation::tracked_descriptor<T>>
                       &input_allocations,
                    sycl::range<Dim> r) {
  return make_async_view(input_allocations, r);
}


}
}

#endif
