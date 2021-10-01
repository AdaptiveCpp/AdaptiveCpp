/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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

#ifndef HIPSYCL_LOCAL_MEMORY_HPP
#define HIPSYCL_LOCAL_MEMORY_HPP

#include <type_traits>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "sp_group.hpp"

namespace hipsycl {
namespace sycl {

template<class T, class SpGroup>
class local_memory
{
public:
  using scalar_type = typename std::remove_extent<T>::type;

  template<class t = scalar_type, 
          std::enable_if_t<std::is_array<T>::value>* = nullptr>
  [[deprecated("Use sycl::memory_environment() instead")]]
  HIPSYCL_KERNEL_TARGET
  scalar_type& operator[](std::size_t index) noexcept{
    return _var[index];
  }

  [[deprecated("Use sycl::memory_environment() instead")]]
  HIPSYCL_KERNEL_TARGET
  T& operator()() noexcept{
    return _var;
  }
private:
  // It is not possible to just mark this member as __shared__
  // here (at least for HIP/CUDA), because member variables
  // cannot be declared in local memory. The clang plugin
  // therefore finds all declarations of type local_memory<T>
  // and puts them in local memory.
  T _var;
};

}
}

#endif
