
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

#ifndef ACPP_PCUDA_RUNTIME_HPP
#define ACPP_PCUDA_RUNTIME_HPP

#include "detail/dim3.hpp"
#include <cstdlib>

#define ACPP_PCUDA_API extern "C"

namespace hipsycl {
namespace rt {
namespace pcuda {
class stream;
class event;
}
}
}

typedef enum pcudaError : int {
  pcudaSuccess = 0,
  pcudaErrorMissingConfiguration,
  pcudaErrorMemoryAllocation,
  pcudaErrorInitializationError,
  pcudaErrorLaunchFailure,
  pcudaErrorPriorLaunchFailure,
  pcudaErrorLaunchTimeout,
  pcudaErrorLaunchOutOfResources,
  pcudaErrorInvalidDeviceFunction,
  pcudaErrorInvalidConfiguration,
  pcudaErrorInvalidDevice,
  pcudaErrorInvalidValue,
  pcudaErrorInvalidPitchValue,
  pcudaErrorInvalidSymbol,
  pcudaErrorMapBufferObjectFailed,
  pcudaErrorUnmapBufferObjectFailed,
  pcudaErrorInvalidHostPointer,
  pcudaErrorInvalidDevicePointer,
  pcudaErrorInvalidTexture,
  pcudaErrorInvalidTextureBinding,
  pcudaErrorInvalidChannelDescriptor,
  pcudaErrorInvalidMemcpyDirection,
  pcudaErrorAddressOfConstant,
  pcudaErrorTextureFetchFailed,
  pcudaErrorTextureNotBound,
  pcudaErrorSynchronizationError,
  pcudaErrorInvalidFilterSetting,
  pcudaErrorInvalidNormSetting,
  pcudaErrorMixedDeviceExecution,
  pcudaErrorCudartUnloading,
  pcudaErrorUnknown,
  pcudaErrorNotYetImplemented,
  pcudaErrorMemoryValueTooLarge,
  pcudaErrorInvalidResourceHandle,
  pcudaErrorNotReady,
  pcudaErrorInsufficientDriver,
  pcudaErrorSetOnActiveProcess,
  pcudaErrorInvalidSurface,
  pcudaErrorNoDevice,
  pcudaErrorECCUncorrectable,
  pcudaErrorSharedObjectSymbolNotFound,
  pcudaErrorSharedObjectInitFailed,
  pcudaErrorUnsupportedLimit,
  pcudaErrorDuplicateVariableName,
  pcudaErrorDuplicateTextureName,
  pcudaErrorDuplicateSurfaceName,
  pcudaErrorDevicesUnavailable,
  pcudaErrorInvalidKernelImage,
  pcudaErrorNoKernelImageForDevice,
  pcudaErrorIncompatibleDriverContext,
  pcudaErrorPeerAccessAlreadyEnabled,
  pcudaErrorPeerAccessNotEnabled,
  pcudaErrorDeviceAlreadyInUse,
  pcudaErrorProfilerDisabled,
  pcudaErrorProfilerNotInitialized,
  pcudaErrorProfilerAlreadyStarted,
  pcudaErrorProfilerAlreadyStopped,
  pcudaErrorStartupFailure,
  pcudaErrorApiFailureBase
} pcudaError_t;

using pcudaStream_t = hipsycl::rt::pcuda::stream*;

#define pcudaMemAttachGlobal 0x01
#define pcudaMemAttachHost 0x02
#define pcudaMemAttachSingle 0x04 

ACPP_PCUDA_API void __pcudaPushCallConfiguration(dim3 grid, dim3 block,
                                                 size_t shared_mem = 0,
                                                 pcudaStream_t stream = nullptr);
ACPP_PCUDA_API pcudaError_t __pcudaKernelCall(const char *kernel_name,
                                              void **args,
                                              std::size_t hcf_object,
                                              void **kernel_specific_storage);

ACPP_PCUDA_API pcudaError_t pcudaDriverGetVersion(int *version);

ACPP_PCUDA_API pcudaError_t pcudaGetDeviceCount(int* count);
ACPP_PCUDA_API pcudaError_t pcudaGetDevice(int* dev);
ACPP_PCUDA_API pcudaError_t pcudaSetDevice(int dev);

ACPP_PCUDA_API pcudaError_t pcudaGetPlatformCount(int* count);
ACPP_PCUDA_API pcudaError_t pcudaGetPlatform(int* platform);
ACPP_PCUDA_API pcudaError_t pcudaSetPlatform(int platform);

ACPP_PCUDA_API pcudaError_t pcudaGetBackendCount(int* count);
ACPP_PCUDA_API pcudaError_t pcudaGetBackend(int* backend);
ACPP_PCUDA_API pcudaError_t pcudaSetBackend(int backend);

ACPP_PCUDA_API pcudaError_t pcudaSetDeviceExt(int backend, int platform, int device);

enum pcudaComputeMode {
  pcudaComputeModeDefault = 0,
  pcudaComputeModeExclusive = 1,
  pcudaComputeModeProhibited = 2,
  pcudaComputeModeExclusiveProcess = 3
};

struct pcudaDeviceProp {
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture2D[2];
  int maxTexture3D[3];
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
};

ACPP_PCUDA_API pcudaError_t pcudaGetDeviceProperties(struct pcudaDeviceProp *prop,
                                                    int device);

ACPP_PCUDA_API pcudaError_t pcudaGetLastError();
ACPP_PCUDA_API pcudaError_t pcudaPeekAtLastError();
ACPP_PCUDA_API const char *pcudaGetErrorName(pcudaError_t error);
ACPP_PCUDA_API const char *pcudaGetErrorString(pcudaError_t error);

ACPP_PCUDA_API pcudaError_t pcudaDeviceSynchronize();
ACPP_PCUDA_API pcudaError_t pcudaThreadSynchronize();

ACPP_PCUDA_API pcudaError_t pcudaAllocateDevice(void** ptr, size_t s);
ACPP_PCUDA_API pcudaError_t pcudaAllocateHost(void** ptr, size_t s);
ACPP_PCUDA_API pcudaError_t pcudaAllocateShared(
    void **ptr, size_t s, unsigned int flags = pcudaMemAttachGlobal);

template<class T>
pcudaError_t pcudaMalloc(T** ptr, size_t s) {
  return pcudaAllocateDevice((void**)ptr, s);
}

template<class T>
pcudaError_t pcudaMallocHost(T** ptr, size_t s) {
  return pcudaAllocateHost((void**)ptr, s);
}

#define pcudaHostAllocDefault 0x00
#define pcudaHostAllocMapped 0x02
#define pcudaHostAllocPortable 0x01
#define pcudaHostAllocWriteCombined 0x04 

template <class T>
pcudaError_t pcudaHostAlloc(T **ptr, size_t s, unsigned int flags) {
  return pcudaAllocateHost((void **)ptr, s);
}

template<class T>
pcudaError_t pcudaMallocManaged(T** ptr, size_t s) {
  return pcudaAllocateShared((void**)ptr, s);
}

ACPP_PCUDA_API pcudaError_t pcudaFree(void* ptr);
ACPP_PCUDA_API pcudaError_t pcudaFreeHost(void* ptr);

// Streams

#define pcudaStreamDefault 0x00
#define pcudaStreamNonBlocking 0x01 

ACPP_PCUDA_API pcudaError_t pcudaStreamCreate(pcudaStream_t *pStream);
ACPP_PCUDA_API pcudaError_t pcudaStreamCreateWithFlags(pcudaStream_t *pStream,
                                                       unsigned int flags);
ACPP_PCUDA_API pcudaError_t pcudaStreamCreateWithPriority(
    pcudaStream_t *pStream, unsigned int flags, int priority);
ACPP_PCUDA_API pcudaError_t pcudaStreamDestroy(pcudaStream_t stream);
ACPP_PCUDA_API pcudaError_t pcudaStreamSynchronize(pcudaStream_t stream);

enum pcudaMemcpyKind {
  pcudaMemcpyHostToHost,
  pcudaMemcpyHostToDevice,
  pcudaMemcpyDeviceToHost,
  pcudaMemcpyDeviceToDevice,
  pcudaMemcpyDefault
};

// Data transfer

ACPP_PCUDA_API pcudaError_t pcudaMemcpy(void *dst, const void *src,
                                        size_t count, pcudaMemcpyKind kind);
ACPP_PCUDA_API pcudaError_t pcudaMemcpyAsync(void *dst, const void *src,
                                             size_t count, pcudaMemcpyKind kind,
                                             pcudaStream_t stream = 0);

ACPP_PCUDA_API pcudaError_t pcudaMemset(void *ptr, int value, size_t count);

ACPP_PCUDA_API pcudaError_t pcudaMemsetAsync(void *ptr, int value, size_t count,
                                             pcudaStream_t stream = 0);
// Event

using pcudaEvent_t = hipsycl::rt::pcuda::event *;

ACPP_PCUDA_API pcudaError_t pcudaEventCreate(pcudaEvent_t *event);
ACPP_PCUDA_API pcudaError_t pcudaEventCreateWithFlags(pcudaEvent_t *event,
                                                     unsigned int flags);

#define pcudaEventBlockingSync 0x01
#define pcudaEventDefault 0x00
#define pcudaEventDisableTiming 0x02
#define pcudaEventInterprocess 0x04

#define pcudaEventRecordDefault 0x00
#define pcudaEventRecordExternal 0x01

#define pcudaEventWaitDefault 0x00
#define pcudaEventWaitExternal 0x01

ACPP_PCUDA_API pcudaError_t pcudaEventDestroy(pcudaEvent_t event);
// TBD cudaError_t cudaEventElapsedTime ( float* ms, cudaEvent_t start,
// cudaEvent_t end )
// Returns success if complete, or not ready otherwise
ACPP_PCUDA_API pcudaError_t pcudaEventQuery(pcudaEvent_t event);

ACPP_PCUDA_API pcudaError_t pcudaEventRecord(pcudaEvent_t event,
                                             pcudaStream_t stream = 0);
ACPP_PCUDA_API pcudaError_t pcudaEventRecordWithFlags(pcudaEvent_t event,
                                                      pcudaStream_t stream = 0,
                                                      unsigned int flags = 0);
ACPP_PCUDA_API pcudaError_t pcudaEventSynchronize(pcudaEvent_t event);

ACPP_PCUDA_API pcudaError_t pcudaStreamWaitEvent(pcudaStream_t stream,
                                                 pcudaEvent_t event,
                                                 unsigned int flags = 0);

#endif
