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

#include <cassert>
#include <atomic>
#include <string_view>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"


namespace hipsycl::rt::pcuda {

namespace {

const hardware_context* get_current_device_ctx(){
  int b = pcuda_application::get().tls_state().get_backend();
  int p = pcuda_application::get().tls_state().get_platform();
  int d = pcuda_application::get().tls_state().get_device();

  auto* dev = pcuda_application::get()
      .pcuda_rt()
      .get_topology()
      .get_device(b, p, d);
  if(!dev)
    return nullptr;
  return dev->dev;
}

const device_id* get_current_device_id(){
  int b = pcuda_application::get().tls_state().get_backend();
  int p = pcuda_application::get().tls_state().get_platform();
  int d = pcuda_application::get().tls_state().get_device();

  auto* dev = pcuda_application::get()
      .pcuda_rt()
      .get_topology()
      .get_device(b, p, d);
  if(!dev)
    return nullptr;
  return &(dev->rt_device_id);
}


pcudaStream_t stream_or_default_stream(pcudaStream_t stream) {
  if(!stream) {
    return pcuda_application::get().tls_state().get_default_stream();
  } else {
    return stream;
  }
}

inorder_queue* queue_or_default_queue(pcudaStream_t stream) {
  return stream_get(stream_or_default_stream(stream));
}

auto dim3_size(dim3 v){
  return v.x * v.y * v.z;
}

auto dim3_to_range3(dim3 v) {
  return rt::range<3>{v.x, v.y, v.z};
}

#define return_if_prior_error()                                                \
  pcudaError_t prior_err = get_most_recent_pcuda_error();                      \
  if (prior_err != pcudaSuccess) {                                             \
    return prior_err;                                                          \
  }

const hcf_kernel_info* extract_kernel_info(hcf_object_id id, std::string_view kernel_name, void** kernel_specific_storage) {
  assert(kernel_specific_storage);
  // atomic_ref is C++20 :(
  void* stored_ptr = __atomic_load_n(kernel_specific_storage, __ATOMIC_RELAXED);
  if(stored_ptr)
    return static_cast<hcf_kernel_info*>(stored_ptr);
  else {
    auto* info = hcf_cache::get().get_kernel_info(id, kernel_name);

    __atomic_store_n(kernel_specific_storage,
                     const_cast<void *>(static_cast<const void *>(info)),
                     __ATOMIC_RELAXED);
    return info;
  }
}

}

ACPP_PCUDA_API void __pcudaPushCallConfiguration(dim3 grid, dim3 block,
                                                 size_t shared_mem = 0,
                                                 pcudaStream_t stream = nullptr) {
  thread_local_state::kernel_call_configuration call_config;
  call_config.stream = stream_or_default_stream(stream);
  call_config.grid = grid;
  call_config.block = block;
  call_config.shared_mem = shared_mem;
  pcuda_application::get().tls_state().push_kernel_call_config(call_config);
}

ACPP_PCUDA_API pcudaError_t __pcudaKernelCall(const char *kernel_name,
                                              void **args,
                                              hcf_object_id hcf_object,
                                              void **kernel_specific_storage){
  return_if_prior_error();

  thread_local_state::kernel_call_configuration call_config =
      pcuda_application::get().tls_state().pop_kernel_call_config();
  if(dim3_size(call_config.block) == 0 || dim3_size(call_config.grid) == 0) {
    register_pcuda_error(__acpp_here(), pcudaErrorInvalidConfiguration,
                     "pcudaKernelCall: Grid or block size is 0");
    return pcudaErrorInvalidConfiguration;
  }

  inorder_queue *q = queue_or_default_queue(call_config.stream);
  if(!q)
    return pcudaErrorInvalidValue;

  std::string_view kernel_name_view = std::string_view{kernel_name};
  const rt::hcf_kernel_info *kinfo = extract_kernel_info(
      hcf_object, kernel_name_view, kernel_specific_storage);
  std::size_t num_args = kinfo->get_host_side_parameter_sizes().size();
  // empty config is fine; we don't expect user interaction
  rt::kernel_configuration config;

  result err = q->submit_sscp_kernel_from_code_object(
      hcf_object, kernel_name_view, kinfo, dim3_to_range3(call_config.grid),
      dim3_to_range3(call_config.block), call_config.shared_mem, args,
      const_cast<std::size_t *>(kinfo->get_host_side_parameter_sizes().data()),
      num_args, config);

  if(err.is_success()) {
    return pcudaSuccess;
  } else {
    register_pcuda_error(err, pcudaErrorLaunchFailure);
    return pcudaErrorLaunchFailure;
  }
}

///////////////////// Device management //////////////////

ACPP_PCUDA_API pcudaError_t pcudaGetDeviceCount(int *count) {
  return_if_prior_error();

  if(!count)
    return pcudaErrorInvalidValue;
    

  int b = pcuda_application::get().tls_state().get_backend();
  int p = pcuda_application::get().tls_state().get_platform();
  auto *platform =
      pcuda_application::get().pcuda_rt().get_topology().get_platform(b, p);
  if(!platform)
    return pcudaErrorNoDevice;
  int n = platform->devices.size();
  *count = n;

  if(n == 0)
    return pcudaErrorNoDevice;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetPlatformCount(int *count) {
  return_if_prior_error();

  if(!count)
    return pcudaErrorInvalidValue;

  int b = pcuda_application::get().tls_state().get_backend();
  int n = static_cast<int>(pcuda_application::get()
                                   .pcuda_rt()
                                   .get_topology()
                                   .get_backend(b)->platforms.size());
  *count = n;

  if(n == 0)
    return pcudaErrorNoDevice;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetBackendCount(int *count) {
  return_if_prior_error();

  if(!count)
    return pcudaErrorInvalidValue;

  int n = static_cast<int>(pcuda_application::get()
                                   .pcuda_rt()
                                   .get_topology()
                                   .all_backends().size());
  *count = n;

  if(n == 0)
    return pcudaErrorNoDevice;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetDevice(int *d) {
  return_if_prior_error();

    if(!d)
      return pcudaErrorInvalidValue;

  *d = pcuda_application::get().tls_state().get_device();
  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetPlatform(int *p) {
  return_if_prior_error();

  if(!p)
      return pcudaErrorInvalidValue;
  
  *p = pcuda_application::get().tls_state().get_platform();
  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetBackend(int *b) {
  return_if_prior_error();

  if(!b)
    return pcudaErrorInvalidValue;

  *b = pcuda_application::get().tls_state().get_backend();
  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaSetDevice(int val) {
  return_if_prior_error();

  if(pcuda_application::get().tls_state().set_device(val))
    return pcudaSuccess;
  return pcudaErrorInvalidDevice;
}

ACPP_PCUDA_API pcudaError_t pcudaSetPlatform(int val) {
  return_if_prior_error();

  
  if(pcuda_application::get().tls_state().set_platform(val))
    return pcuda_application::get().tls_state().set_device(0)
               ? pcudaSuccess
               : pcudaErrorNoDevice;

  return pcudaErrorInvalidValue;
}

ACPP_PCUDA_API pcudaError_t pcudaSetBackend(int val) {
  return_if_prior_error();

  if(pcuda_application::get().tls_state().set_backend(val)) {
    return pcudaSetPlatform(0);
  }
  
  return pcudaErrorInvalidValue;
}

///////////// Device synchronization ///////////////////

ACPP_PCUDA_API pcudaError_t pcudaDeviceSynchronize() {
  return_if_prior_error();

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;
  return stream_wait_all(*dev);
}

///////////// Error management /////////////////////////

ACPP_PCUDA_API pcudaError_t pcudaGetLastError() {
  return pop_most_recent_pcuda_error();
}

///////////// Memory management ///////////////////////

ACPP_PCUDA_API pcudaError_t pcudaAllocateDevice(void** ptr, size_t s) {
  return_if_prior_error();

  if(!ptr)
    return pcudaErrorInvalidValue;

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;

  auto* allocator = pcuda_application::get()
      .pcuda_rt()
      .get_rt()
      ->backends()
      .get(dev->get_backend())
      ->get_allocator(*dev);
  
  void* mem = allocate_device(allocator, 0, s, {});
  if(!mem)
    return pcudaErrorMemoryAllocation;
  *ptr = mem;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaAllocateHost(void** ptr, size_t s) {
  return_if_prior_error();

  if(!ptr)
    return pcudaErrorInvalidValue;

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;

  auto* allocator = pcuda_application::get()
      .pcuda_rt()
      .get_rt()
      ->backends()
      .get(dev->get_backend())
      ->get_allocator(*dev);
  
  void* mem = allocate_host(allocator, 0, s, {});
  if(!mem)
    return pcudaErrorMemoryAllocation;
  *ptr = mem;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaAllocateShared(void **ptr, size_t s,
                                               unsigned int flags) {
  return_if_prior_error();

  if(!ptr)
    return pcudaErrorInvalidValue;

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;

  auto* allocator = pcuda_application::get()
      .pcuda_rt()
      .get_rt()
      ->backends()
      .get(dev->get_backend())
      ->get_allocator(*dev);
  
  void* mem = allocate_shared(allocator, s, {});
  if(!mem)
    return pcudaErrorMemoryAllocation;
  *ptr = mem;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaFree(void* ptr) {
  return_if_prior_error();

  // CUDA accepts deallocating nullptr
  if(!ptr)
    return pcudaSuccess;

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;
  auto* allocator = pcuda_application::get()
      .pcuda_rt()
      .get_rt()
      ->backends()
      .get(dev->get_backend())
      ->get_allocator(*dev);
  deallocate(allocator, ptr);

  return pcudaSuccess;
}


}