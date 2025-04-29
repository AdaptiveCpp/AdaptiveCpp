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
#ifndef HIPSYCL_MEM_FENCE_HPP
#define HIPSYCL_MEM_FENCE_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/access.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

template<access::fence_space, access::mode>
struct mem_fence_impl
{
  ACPP_KERNEL_TARGET
  static void mem_fence()
  {

    __acpp_if_target_hiplike(
      __threadfence();
    );
    // TODO What about CPU?
    // Empty __acpp_if_target_* breaks at compile time w/ nvc++ 22.7 or
    // older, so comment out that statement for now.
    //__acpp_if_target_host(/* todo */);
  }

};

template<access::mode M>
struct mem_fence_impl<access::fence_space::local_space, M>
{
  ACPP_KERNEL_TARGET
  static void mem_fence()
  {
    __acpp_if_target_hiplike(
      __threadfence_block();
    );
  }
};



template <
  access::fence_space Fence_space = access::fence_space::global_and_local,
  access::mode Mode = access::mode::read_write
>
ACPP_KERNEL_TARGET
inline void mem_fence()
{
  static_assert(Mode == access::mode::read ||
                Mode == access::mode::write ||
                Mode == access::mode::read_write,
                "mem_fence() is only allowed for read, write "
                "or read_write access modes.");
  mem_fence_impl<Fence_space, Mode>::mem_fence();
}

template<access::mode Mode>
ACPP_KERNEL_TARGET
inline void mem_fence(access::fence_space space)
{
  if(space == access::fence_space::local_space)
    mem_fence<access::fence_space::local_space, Mode>();

  else if(space == access::fence_space::global_space)
    mem_fence<access::fence_space::global_space, Mode>();

  else if(space == access::fence_space::global_and_local)
    mem_fence<access::fence_space::global_and_local, Mode>();
}

}
}
}

#endif
