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
#ifndef HIPSYCL_PCUDA_INTEROP_HPP
#define HIPSYCL_PCUDA_INTEROP_HPP

#include "libkernel/backend.hpp"

#if !ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA &&                                  \
    !ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP && defined(__ACPP_PCUDA__)

#include <cassert>

#include "libkernel/nd_item.hpp"
#include "libkernel/detail/thread_hierarchy.hpp"
#include "property.hpp"
#include "queue.hpp"
#include "device.hpp"

#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"


namespace hipsycl {
namespace sycl {

namespace AdaptiveCpp_pcuda {

template<int D>
nd_item<D> this_nd_item() {
  const sycl::id<D> zero_offset{};
  sycl::nd_item<D> this_item{&zero_offset, sycl::detail::get_group_id<D>(),
                             sycl::detail::get_local_id<D>(),
                             sycl::detail::get_local_size<D>(),
                             sycl::detail::get_grid_size<D>()};
  return this_item;
}

inline sycl::queue make_queue(rt::pcuda::stream* stream) {
  if(!stream)
    stream = rt::pcuda::pcuda_application::get().tls_state().get_default_stream();
  assert(stream);

  rt::device_id dev = rt::pcuda::stream::get_queue(stream)->get_device();

  sycl::property_list props {
    sycl::property::queue::in_order{},
    sycl::property::queue::AdaptiveCpp_inorder_executor{stream->get_executor()}
  };
  return sycl::queue{sycl::device{dev}, props};
}

}

}
}

#endif
#endif
