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
#include <memory>
#include <string_view>

#include "hipSYCL/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/allocator.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/hardware.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_error.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_runtime.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_stream.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_thread_state.hpp"
#include "hipSYCL/runtime/pcuda/pcuda_event.hpp"
#include "hipSYCL/runtime/util.hpp"


namespace hipsycl::rt::pcuda {

namespace {

device_id get_host_device() {
  return device_id{
      backend_descriptor(hardware_platform::cpu, api_platform::omp), 0};
}

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
  pcudaStream_t s = stream_or_default_stream(stream);
  if(!s)
    return nullptr;
  return stream::get_queue(s);
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

ACPP_PCUDA_API pcudaError_t pcudaSetDeviceExt(int backend, int platform, int device) {
  return_if_prior_error();

  pcudaError_t err = pcudaSetBackend(backend);
  if(err != pcudaSuccess)
    return err;

  err = pcudaSetPlatform(platform);
  if(err != pcudaSuccess)
    return err;

  err = pcudaSetDevice(device);
  if(err != pcudaSuccess)
    return err;

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaGetDeviceProperties(struct pcudaDeviceProp *prop,
                                                    int device) {
  return_if_prior_error();

  int b = pcuda_application::get().tls_state().get_backend();
  int p = pcuda_application::get().tls_state().get_platform();
  
  auto* dev = pcuda_application::get()
      .pcuda_rt()
      .get_topology()
      .get_device(b, p, device);
  
  if(!dev)
    return  pcudaErrorInvalidDevice;

  auto string_to_cstring = [](const std::string &s, std::size_t buffer_size,
                              char *buffer) {
    if(buffer_size == 0)
      return;
    for (std::size_t i = 0; (i < s.size()) && (i < buffer_size); ++i) {
      buffer[i] = s[i];
    }
    std::size_t end_pos = std::min(buffer_size - 1, s.size());
    buffer[end_pos] = '\0';
  };

  auto *device_ctx = dev->dev;
  string_to_cstring(device_ctx->get_device_name(), 256, prop->name);

  prop->totalGlobalMem =
      device_ctx->get_property(device_uint_property::global_mem_size);
  prop->sharedMemPerBlock =
      device_ctx->get_property(device_uint_property::local_mem_size);
  prop->regsPerBlock = 0;
  // This is not very useful in case the device supports more than one
  // subgroup size :(
  prop->warpSize =
      device_ctx->get_property(device_uint_list_property::sub_group_sizes)[0];
  
  prop->memPitch = 0;
  prop->maxThreadsPerBlock =
      device_ctx->get_property(device_uint_property::max_group_size);
  prop->maxThreadsDim[0] =
      device_ctx->get_property(device_uint_property::max_group_size0);
  prop->maxThreadsDim[1] =
      device_ctx->get_property(device_uint_property::max_group_size1);
  prop->maxThreadsDim[2] =
      device_ctx->get_property(device_uint_property::max_group_size2);
  
  prop->maxGridSize[0] = std::numeric_limits<int>::max(); // TODO
  prop->maxGridSize[1] = std::numeric_limits<int>::max(); // TODO
  prop->maxGridSize[2] = std::numeric_limits<int>::max(); // TODO

  prop->clockRate =
      device_ctx->get_property(device_uint_property::max_clock_speed);
  prop->totalConstMem = 0;
  prop->major = 0; // ??? How to map these?
  prop->minor = 0;
  prop->textureAlignment = 0;
  prop->deviceOverlap = 1; // TODO
  prop->multiProcessorCount =
      device_ctx->get_property(device_uint_property::max_compute_units);
  prop->kernelExecTimeoutEnabled = 0;
  prop->integrated = 0; // TODO
  prop->canMapHostMemory = 1;
  prop->computeMode = pcudaComputeModeDefault;
  prop->maxTexture1D = 0;
  prop->maxTexture2D[0] = 0;
  prop->maxTexture2D[1] = 0;
  prop->maxTexture3D[0] = 0;
  prop->maxTexture3D[1] = 0;
  prop->maxTexture3D[2] = 0;
  prop->maxTexture1DLayered[0] = 0;
  prop->maxTexture1DLayered[1] = 0;
  prop->maxTexture2DLayered[0] = 0;
  prop->maxTexture2DLayered[1] = 0;
  prop->maxTexture2DLayered[2] = 0;
  prop->surfaceAlignment = 0;
  prop->concurrentKernels = device_ctx->get_max_kernel_concurrency() > 1;
  prop->ECCEnabled = 0; // TODO
  prop->pciBusID = 0;
  prop->pciDeviceID = 0;
  prop->pciDomainID = 0;
  prop->tccDriver = 0;
  // this is not really a correct mapping...
  prop->asyncEngineCount = device_ctx->get_max_memcpy_concurrency();
  prop->unifiedAddressing = 1;
  prop->memoryClockRate = 0;
  prop->memoryBusWidth = 0;
  prop->l2CacheSize =
      device_ctx->get_property(device_uint_property::global_mem_cache_size);
  prop->maxThreadsPerMultiProcessor =
      device_ctx->get_property(device_uint_property::max_group_size);
  return pcudaSuccess;
}

///////////// Device synchronization ///////////////////

ACPP_PCUDA_API pcudaError_t pcudaDeviceSynchronize() {
  return_if_prior_error();

  auto* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;
  return stream::wait_all(*dev);
}

ACPP_PCUDA_API pcudaError_t pcudaThreadSynchronize() {
  return pcudaDeviceSynchronize();
}

///////////// Error management /////////////////////////

ACPP_PCUDA_API pcudaError_t pcudaGetLastError() {
  return pop_most_recent_pcuda_error();
}

ACPP_PCUDA_API pcudaError_t pcudaPeekAtLastError() {
  return get_most_recent_pcuda_error();
}

ACPP_PCUDA_API const char *pcudaGetErrorName(pcudaError_t error) {
  #define DECLARE_ERROR_NAME(errortype) {errortype, #errortype}

  static std::unordered_map<pcudaError_t, std::string> errors = {
    DECLARE_ERROR_NAME(pcudaSuccess),
    DECLARE_ERROR_NAME(pcudaErrorMissingConfiguration),
    DECLARE_ERROR_NAME(pcudaErrorMemoryAllocation),
    DECLARE_ERROR_NAME(pcudaErrorInitializationError),
    DECLARE_ERROR_NAME(pcudaErrorLaunchFailure),
    DECLARE_ERROR_NAME(pcudaErrorPriorLaunchFailure),
    DECLARE_ERROR_NAME(pcudaErrorLaunchTimeout),
    DECLARE_ERROR_NAME(pcudaErrorLaunchOutOfResources),
    DECLARE_ERROR_NAME(pcudaErrorInvalidDeviceFunction),
    DECLARE_ERROR_NAME(pcudaErrorInvalidConfiguration),
    DECLARE_ERROR_NAME(pcudaErrorInvalidDevice),
    DECLARE_ERROR_NAME(pcudaErrorInvalidValue),
    DECLARE_ERROR_NAME(pcudaErrorInvalidPitchValue),
    DECLARE_ERROR_NAME(pcudaErrorInvalidSymbol),
    DECLARE_ERROR_NAME(pcudaErrorMapBufferObjectFailed),
    DECLARE_ERROR_NAME(pcudaErrorUnmapBufferObjectFailed),
    DECLARE_ERROR_NAME(pcudaErrorInvalidHostPointer),
    DECLARE_ERROR_NAME(pcudaErrorInvalidDevicePointer),
    DECLARE_ERROR_NAME(pcudaErrorInvalidTexture),
    DECLARE_ERROR_NAME(pcudaErrorInvalidTextureBinding),
    DECLARE_ERROR_NAME(pcudaErrorInvalidChannelDescriptor),
    DECLARE_ERROR_NAME(pcudaErrorInvalidMemcpyDirection),
    DECLARE_ERROR_NAME(pcudaErrorAddressOfConstant),
    DECLARE_ERROR_NAME(pcudaErrorTextureFetchFailed),
    DECLARE_ERROR_NAME(pcudaErrorTextureNotBound),
    DECLARE_ERROR_NAME(pcudaErrorSynchronizationError),
    DECLARE_ERROR_NAME(pcudaErrorInvalidFilterSetting),
    DECLARE_ERROR_NAME(pcudaErrorInvalidNormSetting),
    DECLARE_ERROR_NAME(pcudaErrorMixedDeviceExecution),
    DECLARE_ERROR_NAME(pcudaErrorCudartUnloading),
    DECLARE_ERROR_NAME(pcudaErrorUnknown),
    DECLARE_ERROR_NAME(pcudaErrorNotYetImplemented),
    DECLARE_ERROR_NAME(pcudaErrorMemoryValueTooLarge),
    DECLARE_ERROR_NAME(pcudaErrorInvalidResourceHandle),
    DECLARE_ERROR_NAME(pcudaErrorNotReady),
    DECLARE_ERROR_NAME(pcudaErrorInsufficientDriver),
    DECLARE_ERROR_NAME(pcudaErrorSetOnActiveProcess),
    DECLARE_ERROR_NAME(pcudaErrorInvalidSurface),
    DECLARE_ERROR_NAME(pcudaErrorNoDevice),
    DECLARE_ERROR_NAME(pcudaErrorECCUncorrectable),
    DECLARE_ERROR_NAME(pcudaErrorSharedObjectSymbolNotFound),
    DECLARE_ERROR_NAME(pcudaErrorSharedObjectInitFailed),
    DECLARE_ERROR_NAME(pcudaErrorUnsupportedLimit),
    DECLARE_ERROR_NAME(pcudaErrorDuplicateVariableName),
    DECLARE_ERROR_NAME(pcudaErrorDuplicateTextureName),
    DECLARE_ERROR_NAME(pcudaErrorDuplicateSurfaceName),
    DECLARE_ERROR_NAME(pcudaErrorDevicesUnavailable),
    DECLARE_ERROR_NAME(pcudaErrorInvalidKernelImage),
    DECLARE_ERROR_NAME(pcudaErrorNoKernelImageForDevice),
    DECLARE_ERROR_NAME(pcudaErrorIncompatibleDriverContext),
    DECLARE_ERROR_NAME(pcudaErrorPeerAccessAlreadyEnabled),
    DECLARE_ERROR_NAME(pcudaErrorPeerAccessNotEnabled),
    DECLARE_ERROR_NAME(pcudaErrorDeviceAlreadyInUse),
    DECLARE_ERROR_NAME(pcudaErrorProfilerDisabled),
    DECLARE_ERROR_NAME(pcudaErrorProfilerNotInitialized),
    DECLARE_ERROR_NAME(pcudaErrorProfilerAlreadyStarted),
    DECLARE_ERROR_NAME(pcudaErrorProfilerAlreadyStopped),
    DECLARE_ERROR_NAME(pcudaErrorStartupFailure),
    DECLARE_ERROR_NAME(pcudaErrorApiFailureBase)
  };

  static const char* not_found = "unrecognized error code";

  auto it = errors.find(error);
  if(it == errors.end())
    return not_found;
  return it->second.c_str();

  #undef DECLARE_ERROR_NAME
}

ACPP_PCUDA_API const char *pcudaGetErrorString(pcudaError_t error) {
  // TODO: Return actual description
  return pcudaGetErrorName(error);
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

ACPP_PCUDA_API pcudaError_t pcudaFreeHost(void* ptr) {
  return_if_prior_error();

  return pcudaFree(ptr);
}


ACPP_PCUDA_API pcudaError_t pcudaStreamCreate(pcudaStream_t *stream) {
  return_if_prior_error();
  return pcudaStreamCreateWithFlags(stream, pcudaStreamDefault);
}

ACPP_PCUDA_API pcudaError_t pcudaStreamCreateWithFlags(pcudaStream_t *stream,
                                                       unsigned int flags) {
  return_if_prior_error();
  return pcudaStreamCreateWithPriority(stream, flags, 0);
}

ACPP_PCUDA_API pcudaError_t pcudaStreamCreateWithPriority(
    pcudaStream_t *stream, unsigned int flags, int priority) {
  return_if_prior_error();

  if(!stream)
    return pcudaErrorInvalidValue;

  if(flags != pcudaStreamDefault && flags != pcudaStreamNonBlocking)
    return pcudaErrorInvalidValue;

  const device_id* dev = get_current_device_id();
  if(!dev)
    return pcudaErrorNoDevice;

  pcuda::stream* s;
  auto err = stream::create(s, &(pcuda_application::get().pcuda_rt()), *dev,
                           flags, priority);
  if(err == pcudaSuccess)
    *stream = s;
  return err;
}

ACPP_PCUDA_API pcudaError_t pcudaStreamDestroy(pcudaStream_t s) {
  return_if_prior_error();

  if(!s)
    return pcudaErrorInvalidValue;
  return stream::destroy(static_cast<pcuda::stream *>(s),
                        &(pcuda_application::get().pcuda_rt()));
}

ACPP_PCUDA_API pcudaError_t pcudaStreamSynchronize(pcudaStream_t stream) {
  return_if_prior_error();

  auto* queue = queue_or_default_queue(stream);
  if(!queue)
    return pcudaErrorNoDevice;
  queue->wait();
  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaMemcpyAsync(void *dst, const void *src,
                                             size_t count, pcudaMemcpyKind kind,
                                             pcudaStream_t stream = 0) {
  return_if_prior_error();

  auto* queue = queue_or_default_queue(stream);
  if(!queue)
    return pcudaErrorNoDevice;

  device_id queue_dev = queue->get_device();
  auto* allocator = pcuda_application::get()
      .pcuda_rt()
      .get_rt()
      ->backends()
      .get(queue_dev.get_backend())
      ->get_allocator(queue_dev);
  assert(allocator);


  auto get_device_for_ptr = [&](const void* ptr) {
    pointer_info info;
    if(!allocator->query_pointer(ptr, info).is_success())
      return get_host_device();
    else {
      if(info.is_optimized_host)
        return get_host_device();
      else if(info.is_usm)
        return queue_dev;
      else
        return device_id{info.dev};
    }
  };

  device_id src_dev = get_device_for_ptr(src);
  device_id dst_dev = get_device_for_ptr(dst);

  memory_location source_location{src_dev, const_cast<void *>(src), rt::id<3>{},
                                  embed_in_range3(range<1>{count}), 1};
  memory_location dest_location{dst_dev, const_cast<void *>(dst), rt::id<3>{},
                                  embed_in_range3(range<1>{count}), 1};

  memcpy_operation op{source_location, dest_location,
                      embed_in_range3(range<1>(count))};

  auto err = queue->submit_memcpy(op, nullptr);
  if(!err.is_success()) {
    register_pcuda_error(err, pcudaErrorUnknown);
    return pcudaErrorUnknown;
  }

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaMemcpy(void *dst, const void *src,
                                        size_t count, pcudaMemcpyKind kind) {
  return_if_prior_error();

  auto err = pcudaMemcpyAsync(dst, src, count, kind, 0);
  if(err != pcudaSuccess)
    return err;

  return pcudaStreamSynchronize(0);
}

ACPP_PCUDA_API pcudaError_t pcudaMemsetAsync(void *ptr, int value, size_t count,
                                             pcudaStream_t stream) {
  return_if_prior_error();

  auto* queue = queue_or_default_queue(stream);
  if(!queue)
    return pcudaErrorNoDevice;

  memset_operation op{ptr, static_cast<unsigned char>(value), count};
  auto err = queue->submit_memset(op, nullptr);

  if(!err.is_success()) {
    register_pcuda_error(err, pcudaErrorUnknown);
  }
  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaMemset(void *ptr, int value, size_t count) {
  return_if_prior_error();
  auto err = pcudaMemsetAsync(ptr, value, count, 0);
  if(err != pcudaSuccess)
    return err;

  return pcudaStreamSynchronize(0);
}

ACPP_PCUDA_API pcudaError_t pcudaEventCreate(pcudaEvent_t *event) {
  return pcudaEventCreateWithFlags(event, 0);
}

ACPP_PCUDA_API pcudaError_t pcudaEventCreateWithFlags(pcudaEvent_t *event, unsigned int flags) {
  return_if_prior_error();
  if(!event)
    return pcudaErrorInvalidValue;

  auto err =
      pcuda::event::create(*event, &pcuda_application::get().pcuda_rt(), flags);

  return err;
}

ACPP_PCUDA_API pcudaError_t pcudaEventDestroy(pcudaEvent_t event) {
  return_if_prior_error();

  if(!event)
    return pcudaErrorInvalidValue;

  return pcuda::event::destroy(event);
}

ACPP_PCUDA_API pcudaError_t pcudaEventQuery(pcudaEvent_t event) {
  return_if_prior_error();

  if(!event)
    return pcudaErrorInvalidValue;

  if(!event->is_recorded())
    return pcudaErrorNotReady;

  return event->is_complete() ? pcudaSuccess : pcudaErrorNotReady;
}

ACPP_PCUDA_API pcudaError_t pcudaEventRecordWithFlags(pcudaEvent_t event,
                                                      pcudaStream_t stream,
                                                      unsigned int flags) {
  return_if_prior_error();

  if(!event)
    return pcudaErrorInvalidValue;

  inorder_queue* q = queue_or_default_queue(stream);
  if(!q)
    return pcudaErrorInvalidResourceHandle;

  return event->record(q);
}

ACPP_PCUDA_API pcudaError_t pcudaEventRecord(pcudaEvent_t event, pcudaStream_t stream) {
  return pcudaEventRecordWithFlags(event, stream, 0);
}

ACPP_PCUDA_API pcudaError_t pcudaEventSynchronize(pcudaEvent_t event) {
  return_if_prior_error();

  if(!event)
    return pcudaErrorInvalidValue;

  return event->wait();
}

ACPP_PCUDA_API pcudaError_t pcudaStreamWaitEvent(pcudaStream_t stream,
                                                 pcudaEvent_t event,
                                                 unsigned int flags = 0) {
  return_if_prior_error();

  inorder_queue* q = queue_or_default_queue(stream);
  if(!q)
    return pcudaErrorInvalidResourceHandle;
  if(!event)
    return pcudaErrorInvalidResourceHandle;


  if(!event->is_recorded())
    return pcudaSuccess;

  // TODO Create new API that does not need dag_node_ptr?
  dag_node_ptr node =
      std::make_shared<dag_node>(execution_hints{}, node_list_t{}, nullptr,
                                 pcuda_application::get().pcuda_rt().get_rt());
  node->mark_submitted(event->get_event_shared_ptr());
  node->assign_to_device(event->get_device());
  
  bool  is_same_platform = true;
  if(event->get_device().get_backend() != q->get_device().get_backend())
    is_same_platform = false;
  else {
    backend *b = pcuda_application::get().pcuda_rt().get_rt()->backends().get(
        event->get_device().get_backend());
    assert(b);

    std::size_t evt_platform_index =
        b->get_hardware_manager()
            ->get_device(event->get_device().get_id())
            ->get_platform_index();
    std::size_t queue_platform_index =
        b->get_hardware_manager()
            ->get_device(q->get_device().get_id())
            ->get_platform_index();
    
    if(evt_platform_index != queue_platform_index)
      is_same_platform = false;
  }
  
  result err;
  if(is_same_platform) {
    err = q->submit_queue_wait_for(node);
  } else {
    err = q->submit_external_wait_for(node);
  } 

  // This should not happen, so let's make a sticky error
  if(!err.is_success()) {
    register_pcuda_error(err, pcudaErrorUnknown);
    return pcudaErrorUnknown;
  }

  return pcudaSuccess;
}

ACPP_PCUDA_API pcudaError_t pcudaDriverGetVersion(int *version) {
  return_if_prior_error();

  if(!version)
    return pcudaErrorInvalidValue;

  *version = 0;
  return pcudaSuccess;
}

}