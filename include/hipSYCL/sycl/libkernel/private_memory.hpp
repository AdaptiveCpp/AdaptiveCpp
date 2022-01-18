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

#ifndef HIPSYCL_PRIVATE_MEMORY_HPP
#define HIPSYCL_PRIVATE_MEMORY_HPP

#include <memory>

#include "group.hpp"
#include "h_item.hpp"

namespace hipsycl {
namespace sycl {

#ifdef SYCL_DEVICE_ONLY

template<typename T, int Dimensions = 1>
class private_memory
{
public:
  HIPSYCL_KERNEL_TARGET
  private_memory(const group<Dimensions>&)
  {}

  HIPSYCL_KERNEL_TARGET
  T& operator()(const h_item<Dimensions>&) noexcept
  {
    return _data;
  }

private:
  T _data;
};

#else

template<typename T, int Dimensions = 1>
class private_memory
{
public:
  HIPSYCL_KERNEL_TARGET
  private_memory(const group<Dimensions>& grp)
  : _data{new T [grp.get_local_range().size()]}
  {}

  HIPSYCL_KERNEL_TARGET
  T& operator()(const h_item<Dimensions>& idx) noexcept
  {
    return get(idx.get_local_id(), idx.get_local_range());
  }

private:
  std::unique_ptr<T []> _data;

  HIPSYCL_KERNEL_TARGET
  T& get(id<Dimensions> id, range<Dimensions> local_range) noexcept
  {
    return _data.get()[detail::linear_id<Dimensions>::get(id, local_range)];
  }
};
#endif

}
}

#endif
