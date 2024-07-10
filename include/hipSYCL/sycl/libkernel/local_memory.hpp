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
  ACPP_KERNEL_TARGET
  scalar_type& operator[](std::size_t index) noexcept{
    return _var[index];
  }

  [[deprecated("Use sycl::memory_environment() instead")]]
  ACPP_KERNEL_TARGET
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
