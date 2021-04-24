/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
                       property::buffer::hipSYCL_write_back_node_group{
                         q.get_info<info::queue::hipSYCL_node_group>()
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
