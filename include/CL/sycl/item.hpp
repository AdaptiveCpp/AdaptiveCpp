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

#ifndef SYCU_ITEM_HPP
#define SYCU_ITEM_HPP

#include "id.hpp"
#include "range.hpp"

#include "backend/backend.hpp"

#include <type_traits>

namespace cl {
namespace sycl {

namespace detail {

template<int dim>
struct item_impl
{
  static __device__ range<dim> get_range();
  __device__ std::size_t get_linear_id();
};

template<>
struct item_impl<1>
{
  static __device__ range<1> get_range()
  {
    return range<1>{hipGridDim_x * hipBlockDim_x};
  }

  __device__ item_impl()
    : global_id{hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x}
  {}


  __device__ std::size_t get_linear_id()
  {
    return global_id[0];
  }

  id<1> global_id;
};


template<>
struct item_impl<2>
{
  static __device__ range<2> get_range()
  {
    return range<2>{hipGridDim_x * hipBlockDim_x,
                    hipGridDim_y * hipBlockDim_y};
  }

  __device__ item_impl()
    : global_id{hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
                hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y}
  {}


  __device__ std::size_t get_linear_id()
  {
    return  global_id[0] * hipGridDim_y * hipBlockDim_y + global_id[1];
  }

  id<2> global_id;
};


template<>
struct item_impl<3>
{
  static __device__ range<3> get_range()
  {
    return range<2>{hipGridDim_x * hipBlockDim_x,
                    hipGridDim_y * hipBlockDim_y,
                    hipGridDim_z * hipBlockDim_z};
  }

  __device__ item_impl()
    : global_id{hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x,
                hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y,
                hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z}
  {}

  __device__ std::size_t get_linear_id()
  {
    return  global_id[0] * hipGridDim_y * hipBlockDim_y * hipGridDim_z * hipBlockDim_z
          + global_id[1] * hipGridDim_z * hipBlockDim_z
          + global_id[2];
  }

  id<3> global_id;
};

template<int dimension, bool with_offset>
struct item_offset_storage
{
  __device__ item_offset_storage(){}
  __device__ item_offset_storage(const id<dimension>&){}
};


template<int dimension>
struct item_offset_storage<dimension,true>
{
  id<dimension> offset;

  __device__ item_offset_storage(){}
  __device__ item_offset_storage(const id<dimension>& my_offset)
    : offset{my_offset}
  {}
};

}

class handler;

template <int dimensions = 1, bool with_offset = true>
struct item
{
private:
  friend class handler;

  __device__ item() {}
  __device__ item(const id<dimensions>& offset)
    : _offset{offset}
  {}

public:
  /* -- common interface members -- */
  __device__ id<dimensions> get_id() const
  { return _impl.global_id; }

  __device__ std::size_t get_id(int dimension) const
  { return _impl.global_id[dimension]; }

  __device__ std::size_t &operator[](int dimension)
  { return _impl.global_id[dimension]; }

  __device__ range<dimensions> get_range() const
  {
    return detail::item_impl<dimensions>::get_range();
  }

  // only available if with_offset is true
  template<bool O = with_offset,
           typename = std::enable_if_t<O == true>>
  __device__ id<dimensions> get_offset() const
  {
    return _offset.offset;
  }

  // only available if with_offset is false
  template<bool O = with_offset,
           typename = std::enable_if_t<O == false>>
  __device__ operator item<dimensions, true>() const
  {
    return item<dimensions, true>();
  }

  __device__ size_t get_linear_id() const
  {
    return _impl.get_linear_id();
  }

private:
  detail::item_impl<dimensions> _impl;
  detail::item_offset_storage<dimensions, with_offset> _offset;
};



}
}

#endif
