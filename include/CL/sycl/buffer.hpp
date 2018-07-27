/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef SYCU_BUFFER_HPP
#define SYCU_BUFFER_HPP

#include <cstddef>

#include "property.hpp"
#include "types.hpp"
#include "context.hpp"
#include "buffer_allocator.hpp"

#include "id.hpp"
#include "range.hpp"

namespace cl {
namespace sycl {
namespace property {
namespace buffer {

class use_host_ptr : public detail::property
{
public:
  use_host_ptr() = default;
};

class use_mutex : public detail::property
{
public:
  use_mutex(mutex_class& ref);
  mutex_class* get_mutex_ptr() const;
};

class context_bound
{
public:
  context_bound(context bound_context)
    : _ctx{bound_context}
  {}

  context get_context() const
  {
    return _ctx;
  }
private:
  context _ctx;
};

} // property
} // buffer


template <typename T, int dimensions = 1,
          typename AllocatorT = cl::sycl::buffer_allocator>
class buffer : public detail::property_carrying_object
{
public:
  using value_type = T;
  using reference = value_type &;
  using const_reference = const value_type &;
  using allocator_type = AllocatorT;

  buffer(const range<dimensions> &bufferRange,
         const property_list &propList = {})
    : detail::property_carrying_object{propList}
  {

  }

  buffer(const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {});

  buffer(T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {});

  buffer(T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {});

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         const property_list &propList = {});

  buffer(const T *hostData, const range<dimensions> &bufferRange,
         AllocatorT allocator, const property_list &propList = {});

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange, AllocatorT allocator,
         const property_list &propList = {});

  buffer(const shared_ptr_class<T> &hostData,
         const range<dimensions> &bufferRange,
         const property_list &propList = {});

  template <class InputIterator>
  buffer<T, 1>(InputIterator first, InputIterator last, AllocatorT allocator,
               const property_list &propList = {});

  template <class InputIterator>
  buffer<T, 1>(InputIterator first, InputIterator last,
               const property_list &propList = {});

  buffer(buffer<T, dimensions, AllocatorT> b, const id<dimensions> &baseIndex,
         const range<dimensions> &subRange);

  /* Available only when: dimensions == 1. */

  /* CL interop is not supported
  buffer(cl_mem clMemObject, const context &syclContext,
         event availableEvent = {});
  */

  /* -- common interface members -- */

  /* -- property interface members -- */

  range<dimensions> get_range() const
  {
    return _range;
  }

  std::size_t get_count() const
  {
    return _range.size();
  }

  std::size_t get_size() const
  {
    return get_count() * sizeof(T);
  }

  AllocatorT get_allocator() const
  {
    return _alloc;
  }

  template <access::mode mode, access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target> get_access(handler &commandGroupHandler);

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer> get_access();

  template <access::mode mode, access::target target = access::target::global_buffer>
  accessor<T, dimensions, mode, target> get_access(
      handler &commandGroupHandler, range<dimensions> accessRange,
      id<dimensions> accessOffset = {});

  template <access::mode mode>
  accessor<T, dimensions, mode, access::target::host_buffer> get_access(
      range<dimensions> accessRange, id<dimensions> accessOffset = {});


  template <typename Destination = std::nullptr_t>
  void set_final_data(Destination finalData = std::nullptr);

  void set_write_back(bool flag = true);

  bool is_sub_buffer() const;


  template <typename ReinterpretT, int ReinterpretDim>
  buffer<ReinterpretT, ReinterpretDim, AllocatorT>

  reinterpret(range<ReinterpretDim> reinterpretRange) const;

  bool operator==(const platform& rhs) const;

  bool operator!=(const platform& rhs) const;

private:
  AllocatorT _alloc;
  range<dimensions> _range;

  template<class T>
  void allocate_buffer(T* host_ptr, std::size_t bytes, bool use_svm = false);
  void deallocate_buffer();
};



} // sycl
} // cl

#endif
