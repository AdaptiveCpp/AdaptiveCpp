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

template <int dimensions = 1>
struct nd_item
{
  __device__
  nd_item() = delete;

  /* -- common interface members -- */

  __device__
  id<dimensions> get_global() const;

  __device__
  size_t get_global(int dimension) const;

  __device__
  size_t get_global_linear_id() const;

  __device__
  id<dimensions> get_local() const
  {
    return id<dimensions>
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
  group<dimensions> get_group() const;

  __device__
  size_t get_group(int dimension) const;

  __device__
  size_t get_group_linear_id() const;

  __device__
  id<dimensions> get_num_groups() const;

  __device__
  size_t get_num_groups(int dimension) const;

  __device__
  range<dimensions> get_global_range() const;

  __device__
  range<dimensions> get_local_range() const;

  __device__
  id<dimensions> get_offset() const;

  __device__
  nd_range<dimensions> get_nd_range() const;

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
    barrier(accessSpace);
  }

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

};

} // namespace sycl
} // namespace cl

#endif
