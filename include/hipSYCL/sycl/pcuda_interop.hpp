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
#include "platform.hpp"
#include "exception.hpp"
#include "event.hpp"

#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_event.hpp"
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

inline
hipsycl::rt::pcuda::stream* make_stream(sycl::queue& q) {
  auto exec = q.AdaptiveCpp_extract_inorder_executor();

  if(!exec) {
    throw exception{
        make_error_code(errc::invalid),
        "Cannot convert SYCL queue to PCUDA stream; queue does not have a "
        "dedicated inorder-executor (Is it an out-of-order queue?)."};
  }

  hipsycl::rt::pcuda::stream* stream;
  if (rt::pcuda::stream::create(stream, exec) != pcudaSuccess) {
    throw exception{
        make_error_code(errc::runtime),
        "PCUDA stream construction from SYCL queue failed."};
  }
  return stream;
}

inline void make_pcuda_device_indices(const device &dev, int &backend_idx,
                                      int &platform_idx, int &device_idx) {
  auto& rt = hipsycl::rt::pcuda::pcuda_application::get().pcuda_rt();
  rt.get_topology().device_id_to_index_triple(
      dev.AdaptiveCpp_device_id(), backend_idx, platform_idx, device_idx);
}

inline device make_sycl_device(int backend_idx, int platform_idx,
                               int device_idx) {
  auto& rt = hipsycl::rt::pcuda::pcuda_application::get().pcuda_rt();
  auto* dev = rt.get_topology().get_device(backend_idx, platform_idx, device_idx);

  if(!dev)
    throw exception{make_error_code(errc::invalid),
                    "Could not construct SYCL device index from PCUDA device "
                    "indices. Are they correct?"};
  return device{dev->rt_device_id};
}

inline pcudaError_t set_pcuda_device(const device& dev) {
  int backend_idx = 0;
  int platform_idx = 0;
  int dev_idx = 0;
  make_pcuda_device_indices(dev, backend_idx, platform_idx, dev_idx);

  return pcudaSetDeviceExt(backend_idx, platform_idx, dev_idx);
}

inline device get_pcuda_device() {
  int backend, platform, dev;
  pcudaGetBackend(&backend);
  pcudaGetPlatform(&platform);
  pcudaGetDevice(&dev);
  return make_sycl_device(backend, platform, dev);
}

inline sycl::event make_event(hipsycl::rt::pcuda::event* evt) {
  if(!evt)
    throw exception{make_error_code(errc::invalid),
                    "Invalid PCUDA event"};
  if(!evt->get_event())
    throw exception{make_error_code(errc::invalid),
                    "Invalid PCUDA event; does not store internal event "
                    "pointer (perhaps pcudaEventRecord has not yet been called?)."};

  auto* acpp_rt = hipsycl::rt::pcuda::pcuda_application::get().pcuda_rt().get_rt();

  auto op = std::make_unique<rt::kernel_operation>(
      "<no-op>", rt::kernel_launcher({}, {}), rt::requirements_list{acpp_rt});
  rt::dag_node_ptr node = std::make_shared<rt::dag_node>(
      rt::execution_hints{}, rt::node_list_t{}, std::move(op),
      acpp_rt);
  node->assign_to_device(evt->get_device());
  node->assign_to_executor(nullptr);
  node->get_execution_hints().set_hint(rt::hints::instant_execution{});
  node->get_operation()->get_instrumentations().mark_set_complete();
  node->mark_submitted(evt->get_event_shared_ptr());

  return sycl::event{node};
}

}

}
}

#endif
#endif
