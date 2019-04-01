/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
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

#ifndef HIPSYCL_ITEM_HPP
#define HIPSYCL_ITEM_HPP

#include "id.hpp"
#include "range.hpp"
#include "detail/thread_hierarchy.hpp"
#include "backend/backend.hpp"

#include <type_traits>

namespace cl {
namespace sycl {

namespace detail {


template<int dim>
struct item_impl
{
  __device__ item_impl(const sycl::range<dim>& global_size)
    : global_id{detail::get_global_id<dim>()}, global_size(global_size)
  {}

  __device__ item_impl(const id<dim>& my_id, const sycl::range<dim>& global_size)
    : global_id{my_id}, global_size(global_size)
  {}

  id<dim> global_id;
  sycl::range<dim> global_size;
};

template<int dimension, bool with_offset>
struct item_offset_storage
{
  __device__ item_offset_storage(){}
  __device__ item_offset_storage(const id<dimension>){}

  __device__ size_t add_offset(size_t id, int) const
  { return id; }

  __device__ id<dimension> add_offset(const id<dimension>& idx) const
  { return idx; }
};


template<int dimension>
struct item_offset_storage<dimension,true>
{
  const id<dimension> offset;

  __device__ item_offset_storage(){}
  __device__ item_offset_storage(const id<dimension>& my_offset)
    : offset{my_offset}
  {}

  __device__ size_t add_offset(size_t id, int dim) const
  { return id + offset.get(dim); }

  __device__ id<dimension> add_offset(const id<dimension>& idx) const
  { return idx + offset; }
};

}


template <int dimensions = 1, bool with_offset = true>
struct item
{
  __device__ item(const detail::item_impl<dimensions>& impl)
    : _impl{impl}
  {}

  // only available if with_offset is true
  template<bool O = with_offset, typename = std::enable_if_t<O == true>>
  __device__ item(const detail::item_impl<dimensions>& impl, const id<dimensions>& offset)
    : _impl{impl}, _offset_storage{offset}
  {}

  /* -- common interface members -- */
  __device__ id<dimensions> get_id() const
  {
    return _offset_storage.add_offset(
          _impl.global_id);
  }

  __device__ size_t get_id(int dimension) const
  {
    return _offset_storage.add_offset(
          _impl.global_id[dimension], dimension);
  }

  __device__ size_t operator[](int dimension)
  {
    return _offset_storage.add_offset(_impl.global_id[dimension], dimension);
  }

  __device__ range<dimensions> get_range() const
  {
    return _impl.global_size;
  }

  __device__ size_t get_range(int dimension) const
  {
    return _impl.global_size[dimension];
  }

  // only available if with_offset is true
  template<bool O = with_offset,
           typename = std::enable_if_t<O == true>>
  __device__ id<dimensions> get_offset() const
  {
    return _offset_storage.offset;
  }

  // only available if with_offset is false
  template<bool O = with_offset,
           typename = std::enable_if_t<O == false>>
  __device__ operator item<dimensions, true>() const
  {
    return item<dimensions, true>(_impl, id<dimensions>{});
  }

  __device__ size_t get_linear_id() const
  {
    return detail::linear_id<dimensions>::get(
          _offset_storage.add_offset(_impl.global_id),
          _impl.global_size);
  }

private:
  detail::item_impl<dimensions> _impl;
  detail::item_offset_storage<dimensions, with_offset> _offset_storage;
};



}
}

#endif
