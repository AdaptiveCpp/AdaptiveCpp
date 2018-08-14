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

static __device__ size_t get_global_id_x()
{
  return hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
}

static __device__ size_t get_global_id_y()
{
  return hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
}

static __device__ size_t get_global_id_z()
{
  return hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
}

static __device__ size_t get_global_size_x()
{
  return hipGridDim_x * hipBlockDim_x;
}

static __device__ size_t get_global_size_y()
{
  return hipGridDim_y * hipBlockDim_y;
}

static __device__ size_t get_global_size_z()
{
  return hipGridDim_z * hipBlockDim_z;
}


static __host__ __device__ size_t get_linear_id(const size_t id_x,
                                                const size_t id_y,
                                                const size_t range_y)
{
  return id_x * range_y + id_y;
}

static __host__ __device__ size_t get_linear_id(const size_t id_x,
                                                const size_t id_y,
                                                const size_t id_z,
                                                const size_t range_y,
                                                const size_t range_z)
{
  return id_x * range_y * range_z + id_y * range_z + id_z;
}

template<int dim>
struct linear_id
{
};

template<>
struct linear_id<1>
{
  static __host__ __device__ size_t get(const id<1>& idx)
  { return idx[0]; }

  static __host__ __device__ size_t get(const id<1>& idx,
                                        const sycl::range<1>& r)
  {
    return get(idx);
  }
};

template<>
struct linear_id<2>
{
  static __host__ __device__ size_t get(const id<2>& idx,
                                        const sycl::range<2>& r)
  {
    return get_linear_id(idx.get(0), idx.get(1), r.get(1));
  }
};

template<>
struct linear_id<3>
{
  static __host__ __device__ size_t get(const id<3>& idx,
                                        const sycl::range<3>& r)
  {
    return get_linear_id(idx.get(0), idx.get(1), idx.get(2), r.get(1), r.get(2));
  }
};

template<int dim>
struct item_impl
{
  static __device__ sycl::range<dim> get_range();
  __device__ size_t get_linear_id();
};

template<>
struct item_impl<1>
{
  static __device__ sycl::range<1> get_range()
  {
    return sycl::range<1>{get_global_size_x()};
  }

  __device__ item_impl()
    : global_id{get_global_id_x()}
  {}


  __device__ size_t get_linear_id() const
  {
    return global_id[0];
  }

  id<1> global_id;
};


template<>
struct item_impl<2>
{
  static __device__ sycl::range<2> get_range()
  {
    return sycl::range<2>{get_global_size_x(),
                    get_global_size_y()};
  }

  __device__ item_impl()
    : global_id{get_global_id_x(),
                get_global_id_y()}
  {}


  __device__ size_t get_linear_id() const
  {
    return detail::get_linear_id(global_id[0], global_id[1],
                                 get_global_size_y());
  }

  id<2> global_id;
};


template<>
struct item_impl<3>
{
  static __device__ sycl::range<3> get_range()
  {
    return sycl::range<3>{
          get_global_size_x(),
          get_global_size_y(),
          get_global_size_z()
    };
  }

  __device__ item_impl()
    : global_id{get_global_id_x(),
                get_global_id_y(),
                get_global_id_z()}
  {}

  __device__ size_t get_linear_id() const
  {
    return detail::get_linear_id(global_id[0], global_id[1], global_id[2],
                                 get_global_size_y(),
                                 get_global_size_z());
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


template <int dimensions = 1, bool with_offset = true>
struct item
{
  __device__ item() {}
  __device__ item(const id<dimensions>& offset)
    : _offset{offset}
  {}


  /* -- common interface members -- */
  __device__ id<dimensions> get_id() const
  { return _impl.global_id; }

  __device__ size_t get_id(int dimension) const
  { return _impl.global_id[dimension]; }

  __device__ size_t &operator[](int dimension)
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
