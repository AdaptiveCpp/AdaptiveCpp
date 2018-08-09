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

#ifndef SYCU_ND_ITEM_HPP
#define SYCU_ND_ITEM_HPP

#include "id.hpp"
#include "item.hpp"
#include "range.hpp"
#include "nd_range.hpp"

namespace cl {
namespace sycl {

namespace detail {

template<int dimensions>
__device__
static id<dimensions> get_local_id();

template<>
__device__
id<1> get_local_id<1>()
{ return id<1>{hipThreadIdx_x}; }

template<>
__device__
id<2> get_local_id<2>()
{ return id<2>{hipThreadIdx_x, hipThreadIdx_y}; }

template<>
__device__
id<3> get_local_id<3>()
{ return id<3>{hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z}; }

template<int dimensions>
__device__
static id<dimensions> get_global_id();

template<>
__device__
id<1> get_global_id<1>()
{ return id<1>{get_global_id_x()}; }

template<>
__device__
id<2> get_global_id<2>()
{
  return id<2>{get_global_id_x(),
               get_global_id_y()};
}

template<>
__device__
id<3> get_global_id<3>()
{
  return id<3>{get_global_id_x(),
               get_global_id_y(),
               get_global_id_z()};
}

}

class handler;

template <int dimensions = 1>
struct nd_item
{
  /* -- common interface members -- */

  __device__
  id<dimensions> get_global() const
  {
    return detail::get_global_id<dimensions>();
  }

  __device__
  size_t get_global(int dimension) const
  {
    switch(dimension)
    {
    case 0:
      return detail::get_global_id_x();
    case 1:
      return detail::get_global_id_y();
    case 2:
      return detail::get_global_id_z();
    }
    return 1;
  }

  __device__
  size_t get_global_linear_id() const;

  __device__
  id<dimensions> get_local() const
  {
    return detail::get_local_id<dimensions>();
  }

  __device__
  size_t get_local(int dimension) const
  {
    switch(dimension)
    {
    case 0:
      return hipThreadIdx_x;
    case 1:
      return hipThreadIdx_y;
    case 2:
      return hipThreadIdx_z;
    }
    return 1;
  }

  __device__
  size_t get_local_linear_id() const;

  __device__
  group<dimensions> get_group() const
  {

  }

  __device__
  size_t get_group(int dimension) const;

  __device__
  size_t get_group_linear_id() const;

  __device__
  id<dimensions> get_num_groups() const
  {
    return _range.get_group();
  }

  __device__
  size_t get_num_groups(int dimension) const
  {
    return _range.get_group().get(dimension);
  }

  __device__
  range<dimensions> get_global_range() const
  {
    return _range.get_global();
  }

  __device__
  range<dimensions> get_local_range() const
  {
    return _range.get_local();
  }

  __device__
  id<dimensions> get_offset() const
  {
    return _range.get_offset();
  }

  __device__
  nd_range<dimensions> get_nd_range() const
  {
    return _range;
  }

  __device__
  void barrier(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    __syncthreads();
  }

  template <access::mode accessMode = access::mode::read_write>
  __device__
  void mem_fence(access::fence_space accessSpace =
      access::fence_space::global_and_local) const
  {
    __syncthreads();
  }

  // ToDo: Implement these
  template <typename dataT>
  __device__
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements) const;

  template <typename dataT>
  __device__
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements) const;

  template <typename dataT>
  __device__
  device_event async_work_group_copy(local_ptr<dataT> dest,
                                     global_ptr<dataT> src, size_t numElements, size_t srcStride) const;

  template <typename dataT>
  __device__
  device_event async_work_group_copy(global_ptr<dataT> dest,
                                     local_ptr<dataT> src, size_t numElements, size_t destStride) const;

  template <typename... eventTN>
  __device__
  void wait_for(eventTN... events) const
  {}

private:
  const nd_range<dimensions> _range;

  __device__
  nd_item(const nd_range<dimensions>& range)
    : _range{range}
  {}


};

} // namespace sycl
} // namespace cl

#endif
