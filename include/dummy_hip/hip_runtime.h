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

#ifndef HIP_DUMMY_RUNTIME_H
#define HIP_DUMMY_RUNTIME_H


#ifndef __global__
#define __global__ __attribute__((target("kernel")))
#endif

#ifndef __device__
#define __device__ __attribute__((target("device")))
#endif

#ifndef __host__
#define __host__ __attribute__((target("host")))
#endif

#ifndef __constant__
#define __constant__
#endif

#ifndef __shared__
#define __shared__
#endif


#include <cstddef>
#include <climits>

#define HIP_KERNEL_NAME(...) __VA_ARGS__

typedef int hipLaunchParm;

#define hipLaunchKernel(kernelName, numblocks, numthreads, memperblock, streamId, ...) \
        kernelName(0, ##__VA_ARGS__);

#define hipLaunchKernelGGL(kernelName, numblocks, numthreads, memperblock, streamId, ...) \
        kernelName(__VA_ARGS__);



#define hipThreadIdx_x 0
#define hipThreadIdx_y 0
#define hipThreadIdx_z 0

#define hipBlockIdx_x 0
#define hipBlockIdx_y 0
#define hipBlockIdx_z 0

#define hipBlockDim_x 1
#define hipBlockDim_y 1
#define hipBlockDim_z 1

#define hipGridDim_x 1
#define hipGridDim_y 1
#define hipGridDim_z 1

#define HIP_SYMBOL(X) X

typedef enum hipMemcpyKind {
    hipMemcpyHostToHost,
    hipMemcpyHostToDevice,
    hipMemcpyDeviceToHost,
    hipMemcpyDeviceToDevice,
    hipMemcpyDefault
} hipMemcpyKind;

// hipTextureAddressMode
#define hipTextureAddressMode 0
#define hipAddressModeWrap 0
#define hipAddressModeClamp 0
#define hipAddressModeMirror 0
#define hipAddressModeBorder 0

// hipTextureFilterMode
#define hipTextureFilterMode 0
#define hipFilterModePoint 0
#define hipFilterModeLinear 0

// hipTextureReadMode
enum hipTextureReadMode {};
#define hipReadModeElementType 0
#define hipReadModeNormalizedFloat 0

template<class T, int dim, hipTextureReadMode readMode>
struct texture {};

typedef enum hipChannelFormatKind {
    hipChannelFormatKindSigned = 0,
    hipChannelFormatKindUnsigned = 1,
    hipChannelFormatKindFloat = 2,
    hipChannelFormatKindNone = 3
} hipChannelFormatKind;

#define hipSurfaceBoundaryMode 0
#define hipBoundaryModeZero 0
#define hipBoundaryModeTrap 0
#define hipBoundaryModeClamp 0

// hipResourceType
#define hipResourceType 0
#define hipResourceTypeArray 0
#define hipResourceTypeMipmappedArray 0
#define hipResourceTypeLinear 0
#define hipResourceTypePitch2D 0

#define hipEventDefault hipEvent_t()
#define hipEventBlockingSync 0
#define hipEventDisableTiming 0
#define hipEventInterprocess 0
#define hipEventReleaseToDevice 0
#define hipEventReleaseToSystem 0


#define hipHostMallocDefault(...)
#define hipHostMallocPortable(...)
#define hipHostMallocMapped(...)
#define hipHostMallocWriteCombined(...)
#define hipHostMallocCoherent(...) 0x0
#define hipHostMallocNonCoherent(...) 0x0

#define hipHostRegisterPortable 0
#define hipHostRegisterMapped 0


typedef int hipEvent_t;
typedef int hipStream_t;
typedef int hipIpcEventHandle_t;
typedef int hipIpcMemHandle_t;
typedef int hipLimit_t;
typedef int hipFuncCache_t;
typedef int hipCtx_t;
typedef int hipSharedMemConfig;
typedef int hipFuncCache;
typedef int hipJitOption;
typedef int hipDevice_t;
typedef int hipModule_t;
typedef int hipFunction_t;
typedef void* hipDeviceptr_t;
typedef int hipArray;
typedef int* hipArray_const_t;
typedef int hipFuncAttributes;
typedef int hipCtx_t;

typedef int hipTextureObject_t;
typedef int hipSurfaceObject_t;
typedef int hipResourceDesc;
typedef int hipTextureDesc;
typedef int hipResourceViewDesc;
typedef int textureReference;

enum hipError_t
{
  hipSuccess,
  hipErrorInvalidContext,
  hipErrorInvalidKernelFile,
  hipErrorMemoryAllocation,
  hipErrorInitializationError,
  hipErrorLaunchFailure,
  hipErrorLaunchOutOfResources,
  hipErrorInvalidDevice,
  hipErrorInvalidValue,
  hipErrorInvalidDevicePointer,
  hipErrorInvalidMemcpyDirection,
  hipErrorUnknown,
  hipErrorInvalidResourceHandle,
  hipErrorNotReady,
  hipErrorNoDevice,
  hipErrorPeerAccessAlreadyEnabled,
  hipErrorPeerAccessNotEnabled,
  hipErrorRuntimeMemory,
  hipErrorRuntimeOther,
  hipErrorHostMemoryAlreadyRegistered,
  hipErrorHostMemoryNotRegistered,
  hipErrorMapBufferObjectFailed,
  hipErrorTbd
};

typedef void* hipPitchedPtr;
struct hipExtent {};

struct hipChannelFormatDesc {};

struct hipDeviceArch_t
{
  unsigned hasGlobalInt32Atomics    : 1;
  unsigned hasGlobalFloatAtomicExch : 1;
  unsigned hasSharedInt32Atomics    : 1;
  unsigned hasSharedFloatAtomicExch : 1;
  unsigned hasFloatAtomicAdd        : 1;

  // 64-bit Atomics
  unsigned hasGlobalInt64Atomics    : 1;
  unsigned hasSharedInt64Atomics    : 1;

  // Doubles
  unsigned hasDoubles               : 1;

  // Warp cross-lane operations
  unsigned hasWarpVote              : 1;
  unsigned hasWarpBallot            : 1;
  unsigned hasWarpShuffle           : 1;
  unsigned hasFunnelShift           : 1;

  // Sync
  unsigned hasThreadFenceSystem     : 1;
  unsigned hasSyncThreadsExt        : 1;

  // Misc
  unsigned hasSurfaceFuncs          : 1;
  unsigned has3dGrid                : 1;
  unsigned hasDynamicParallelism    : 1;
};

struct hipDeviceProp_t
{
  char name[256];
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  int memoryClockRate;
  int memoryBusWidth;
  size_t totalConstMem;
  int major;
  int minor;
  int multiProcessorCount;
  int l2CacheSize;
  int maxThreadsPerMultiProcessor;
  int computeMode;
  int clockInstructionRate;
  hipDeviceArch_t arch;
  int concurrentKernels;
  int pciBusID;
  int pciDeviceID;
  size_t maxSharedMemoryPerMultiProcessor;
  int isMultiGpuBoard;
  int canMapHostMemory;
  int gcnArch;
};

struct dim3
{
  dim3(int d0 = 1, int d1 = 1, int d2 = 1)
    : x{d0}, y{d1}, z{d2}
  {}

  int x;
  int y;
  int z;
};

struct hipMemcpy3DParms {};
enum hipDeviceAttribute_t
{
  hipDeviceAttributeMaxThreadsPerBlock,
  hipDeviceAttributeMaxBlockDimX,
  hipDeviceAttributeMaxBlockDimY,
  hipDeviceAttributeMaxBlockDimZ,
  hipDeviceAttributeMaxGridDimX,
  hipDeviceAttributeMaxGridDimY,
  hipDeviceAttributeMaxGridDimZ,
  hipDeviceAttributeMaxSharedMemoryPerBlock,
  hipDeviceAttributeTotalConstantMemory,
  hipDeviceAttributeWarpSize,
  hipDeviceAttributeMaxRegistersPerBlock,
  hipDeviceAttributeClockRate,
  hipDeviceAttributeMemoryClockRate,
  hipDeviceAttributeMemoryBusWidth,
  hipDeviceAttributeMultiprocessorCount,
  hipDeviceAttributeComputeMode,
  hipDeviceAttributeL2CacheSize,
  hipDeviceAttributeMaxThreadsPerMultiProcessor,
  hipDeviceAttributeComputeCapabilityMajor,
  hipDeviceAttributeComputeCapabilityMinor,
  hipDeviceAttributeConcurrentKernels,
  hipDeviceAttributePciBusId,
  hipDeviceAttributePciDeviceId,
  hipDeviceAttributeMaxSharedMemoryPerMultiprocessor,
  hipDeviceAttributeIsMultiGpuBoard,
  hipDeviceAttributeIntegrated,
};

struct hipPointerAttribute_t
{
  hipDevice_t device;
  hipDeviceptr_t devicePointer;
  void* hostPointer;
  bool isManaged;
  int allocationFlags;
};

#define hipStreamDefault 0
#define hipStreamNonBlocking 0

#define hipSharedMemBankSizeDefault 0
#define hipSharedMemBankSizeFourByte 0
#define hipSharedMemBankSizeEightByte 0

typedef void(*hipStreamCallback_t)(hipStream_t, hipError_t, void*);

void __syncthreads();

hipError_t hipDeviceReset();
hipError_t hipGetLastError();
inline static hipError_t hipPeekAtLastError();
inline static hipError_t hipMalloc(void** ptr, size_t size);
inline static hipError_t hipMallocPitch(void** ptr, size_t* pitch, size_t width, size_t height);
inline static hipError_t hipMalloc3D(hipPitchedPtr* pitchedDevPtr, hipExtent extent);
inline static hipError_t hipFree(void* ptr);
inline static hipError_t hipMallocHost(void** ptr, size_t size);
inline static hipError_t hipHostAlloc(void** ptr, size_t size, unsigned int flags);
inline static hipError_t hipHostMalloc(void** ptr, size_t size, unsigned int flags);
inline static hipError_t hipMallocArray(hipArray** array,
                                        const hipChannelFormatDesc* desc,
                                        size_t width, size_t height,
                                        unsigned int flags);
inline static hipError_t hipMalloc3DArray(hipArray** array, const struct hipChannelFormatDesc* desc,
                            struct hipExtent extent, unsigned int flags);
inline static hipError_t hipFreeArray(hipArray* array);
inline static hipError_t hipHostGetDevicePointer(void** devPtr, void* hostPtr, unsigned int flags);
inline static hipError_t hipHostGetFlags(unsigned int* flagsPtr, void* hostPtr);
inline static hipError_t hipHostRegister(void* ptr, size_t size, unsigned int flags);
inline static hipError_t hipHostUnregister(void* ptr);
inline static hipError_t hipFreeHost(void* ptr);
inline static hipError_t hipHostFree(void* ptr);
inline static hipError_t hipSetDevice(int device);
inline static hipError_t hipChooseDevice(int* device, const hipDeviceProp_t* prop);
inline static hipError_t hipMemcpyHtoD(hipDeviceptr_t dst, void* src, size_t size);
inline static hipError_t hipMemcpyDtoH(void* dst, hipDeviceptr_t src, size_t size);
inline static hipError_t hipMemcpyDtoD(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size);
inline static hipError_t hipMemcpyHtoDAsync(hipDeviceptr_t dst, void* src, size_t size,
                                            hipStream_t stream);
inline static hipError_t hipMemcpyDtoHAsync(void* dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream);
inline static hipError_t hipMemcpyDtoDAsync(hipDeviceptr_t dst, hipDeviceptr_t src, size_t size,
                                            hipStream_t stream);
inline static hipError_t hipMemcpy(void* dst, const void* src, size_t sizeBytes,
                                   hipMemcpyKind copyKind);
inline static hipError_t hipMemcpyAsync(void* dst, const void* src, size_t sizeBytes,
                                        hipMemcpyKind copyKind, hipStream_t stream = 0);
inline static hipError_t hipMemcpyToSymbol(const void* symbol, const void* src, size_t sizeBytes,
                                           size_t offset = 0,
                                           hipMemcpyKind copyType = hipMemcpyHostToDevice);

inline static hipError_t hipMemcpyToSymbolAsync(const void* symbol, const void* src,
                                                size_t sizeBytes, size_t offset,
                                                hipMemcpyKind copyType,
                                                hipStream_t stream = 0);

inline static hipError_t hipMemcpyFromSymbol(void* dst, const void* symbolName, size_t sizeBytes,
                                             size_t offset = 0,
                                             hipMemcpyKind kind = hipMemcpyDeviceToHost);

inline static hipError_t hipMemcpyFromSymbolAsync(void* dst, const void* symbolName,
                                                  size_t sizeBytes, size_t offset,
                                                  hipMemcpyKind kind,
                                                  hipStream_t stream = 0);

inline static hipError_t hipMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch,
                                     size_t width, size_t height, hipMemcpyKind kind);

inline static hipError_t hipMemcpy3D(const struct hipMemcpy3DParms *p);

inline static hipError_t hipMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch,
                                          size_t width, size_t height, hipMemcpyKind kind,
                                          hipStream_t stream);

inline static hipError_t hipMemcpy2DToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                                            const void* src, size_t spitch, size_t width,
                                            size_t height, hipMemcpyKind kind);

inline static hipError_t hipMemcpyToArray(hipArray* dst, size_t wOffset, size_t hOffset,
                                          const void* src, size_t count, hipMemcpyKind kind);

inline static hipError_t hipMemcpyFromArray(void* dst, hipArray_const_t srcArray, size_t wOffset,
                                            size_t hOffset, size_t count, hipMemcpyKind kind);

inline static hipError_t hipMemcpyAtoH(void* dst, hipArray* srcArray, size_t srcOffset,
                                       size_t count);

inline static hipError_t hipMemcpyHtoA(hipArray* dstArray, size_t dstOffset, const void* srcHost,
                                       size_t count);

inline static hipError_t hipDeviceSynchronize();

inline static hipError_t hipDeviceGetCacheConfig(hipFuncCache_t* pCacheConfig);

inline static const char* hipGetErrorString(hipError_t error);

inline static const char* hipGetErrorName(hipError_t error);

inline static hipError_t hipGetDeviceCount(int* count);

inline static hipError_t hipGetDevice(int* device);

inline static hipError_t hipIpcCloseMemHandle(void* devPtr);

inline static hipError_t hipIpcGetEventHandle(hipIpcEventHandle_t* handle, hipEvent_t event);

inline static hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle, void* devPtr);

inline static hipError_t hipIpcOpenEventHandle(hipEvent_t* event, hipIpcEventHandle_t handle);

inline static hipError_t hipIpcOpenMemHandle(void** devPtr, hipIpcMemHandle_t handle,
                                             unsigned int flags);

inline static hipError_t hipMemset(void* devPtr, int value, size_t count);

inline static hipError_t hipMemsetAsync(void* devPtr, int value, size_t count,
                                        hipStream_t stream = 0);

inline static hipError_t hipMemsetD8(hipDeviceptr_t dest, unsigned char value, size_t sizeBytes);

inline static hipError_t hipMemset2D(void* dst, size_t pitch, int value, size_t width, size_t height);

inline static hipError_t hipMemset2DAsync(void* dst, size_t pitch, int value, size_t width, size_t height, hipStream_t stream = 0);

inline static hipError_t hipMemset3D(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent );

inline static hipError_t hipMemset3DAsync(hipPitchedPtr pitchedDevPtr, int  value, hipExtent extent, hipStream_t stream = 0);


inline static hipError_t hipGetDeviceProperties(hipDeviceProp_t* p_prop, int device);

inline static hipError_t hipDeviceGetAttribute(int* pi, hipDeviceAttribute_t attr, int device);

inline static hipError_t hipOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks,
                                                                      const void* func,
                                                                      int blockSize,
                                                                      size_t dynamicSMemSize);

inline static hipError_t hipPointerGetAttributes(hipPointerAttribute_t* attributes, void* ptr);

inline static hipError_t hipMemGetInfo(size_t* free, size_t* total);

inline static hipError_t hipEventCreate(hipEvent_t* event);

inline static hipError_t hipEventRecord(hipEvent_t event, hipStream_t stream = 0);

inline static hipError_t hipEventSynchronize(hipEvent_t event);

inline static hipError_t hipEventElapsedTime(float* ms, hipEvent_t start, hipEvent_t stop);

inline static hipError_t hipEventDestroy(hipEvent_t event);

inline static hipError_t hipStreamCreateWithFlags(hipStream_t* stream, unsigned int flags);

inline static hipError_t hipStreamCreate(hipStream_t* stream);

inline static hipError_t hipStreamSynchronize(hipStream_t stream);

inline static hipError_t hipStreamDestroy(hipStream_t stream);

inline static hipError_t hipStreamWaitEvent(hipStream_t stream, hipEvent_t event,
                                            unsigned int flags);

inline static hipError_t hipStreamQuery(hipStream_t stream);

inline static hipError_t hipStreamAddCallback(hipStream_t stream, hipStreamCallback_t callback,
                                              void* userData, unsigned int flags);

inline static hipError_t hipDriverGetVersion(int* driverVersion);

inline static hipError_t hipRuntimeGetVersion(int* runtimeVersion);

inline static hipError_t hipDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);

inline static hipError_t hipDeviceDisablePeerAccess(int peerDevice);

inline static hipError_t hipDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
inline static hipError_t hipCtxDisablePeerAccess(hipCtx_t peerCtx);

inline static hipError_t hipCtxEnablePeerAccess(hipCtx_t peerCtx, unsigned int flags);

inline static hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags,
                                                     int* active);

inline static hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev);

inline static hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev);

inline static hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev);

inline static hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags);

inline static hipError_t hipMemGetAddressRange(hipDeviceptr_t* pbase, size_t* psize,
                                               hipDeviceptr_t dptr);

inline static hipError_t hipMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice,
                                       size_t count);

inline static hipError_t hipMemcpyPeerAsync(void* dst, int dstDevice, const void* src,
                                            int srcDevice, size_t count,
                                            hipStream_t stream = 0);

// Profile APIs:
inline static hipError_t hipProfilerStart();
inline static hipError_t hipProfilerStop();

inline static hipError_t hipSetDeviceFlags(unsigned int flags);

inline static hipError_t hipEventCreateWithFlags(hipEvent_t* event, unsigned int flags);

inline static hipError_t hipEventQuery(hipEvent_t event);

inline static hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device);

inline static hipError_t hipCtxDestroy(hipCtx_t ctx);

inline static hipError_t hipCtxPopCurrent(hipCtx_t* ctx);

inline static hipError_t hipCtxPushCurrent(hipCtx_t ctx);

inline static hipError_t hipCtxSetCurrent(hipCtx_t ctx);

inline static hipError_t hipCtxGetCurrent(hipCtx_t* ctx);

inline static hipError_t hipCtxGetDevice(hipDevice_t* device);

inline static hipError_t hipCtxGetApiVersion(hipCtx_t ctx, int* apiVersion);

inline static hipError_t hipCtxGetCacheConfig(hipFuncCache* cacheConfig);

inline static hipError_t hipCtxSetCacheConfig(hipFuncCache cacheConfig);

inline static hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config);

inline static hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig);

inline static hipError_t hipCtxSynchronize(void);

inline static hipError_t hipCtxGetFlags(unsigned int* flags);

inline static hipError_t hipCtxDetach(hipCtx_t ctx);

inline static hipError_t hipDeviceGet(hipDevice_t* device, int ordinal);

inline static hipError_t hipDeviceComputeCapability(int* major, int* minor, hipDevice_t device);

inline static hipError_t hipDeviceGetName(char* name, int len, hipDevice_t device);

inline static hipError_t hipDeviceGetPCIBusId(char* pciBusId, int len, hipDevice_t device);

inline static hipError_t hipDeviceGetByPCIBusId(int* device, const char* pciBusId);

inline static hipError_t hipDeviceGetSharedMemConfig(hipSharedMemConfig* config);

inline static hipError_t hipDeviceSetSharedMemConfig(hipSharedMemConfig config);

inline static hipError_t hipDeviceGetLimit(size_t* pValue, hipLimit_t limit);

inline static hipError_t hipDeviceTotalMem(size_t* bytes, hipDevice_t device);

inline static hipError_t hipModuleLoad(hipModule_t* module, const char* fname);

inline static hipError_t hipModuleUnload(hipModule_t hmod);

inline static hipError_t hipModuleGetFunction(hipFunction_t* function, hipModule_t module,
                                              const char* kname);

inline static hipError_t hipFuncGetAttributes(hipFuncAttributes* attr, const void* func);

inline static hipError_t hipModuleGetGlobal(hipDeviceptr_t* dptr, size_t* bytes, hipModule_t hmod,
                                            const char* name);

inline static hipError_t hipModuleLoadData(hipModule_t* module, const void* image);

inline static hipError_t hipModuleLoadDataEx(hipModule_t* module, const void* image,
                                             unsigned int numOptions, hipJitOption* options,
                                             void** optionValues);

inline static hipError_t hipModuleLaunchKernel(hipFunction_t f, unsigned int gridDimX,
                                               unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY,
                                               unsigned int blockDimZ, unsigned int sharedMemBytes,
                                               hipStream_t stream, void** kernelParams,
                                               void** extra);


inline static hipError_t hipFuncSetCacheConfig(const void* func, hipFuncCache_t cacheConfig);


template <class T>
inline static hipError_t hipOccupancyMaxPotentialBlockSize(int* minGridSize, int* blockSize, T func,
                                                           size_t dynamicSMemSize = 0,
                                                           int blockSizeLimit = 0,
                                                           unsigned int flags = 0);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, const struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, size_t size = UINT_MAX);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTexture(size_t* offset, struct texture<T, dim, readMode>& tex,
                                        const void* devPtr, const struct hipChannelFormatDesc& desc,
                                        size_t size = UINT_MAX);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipUnbindTexture(struct texture<T, dim, readMode>* tex);

inline static hipError_t hipBindTexture(size_t* offset, textureReference* tex, const void* devPtr,
                                        const hipChannelFormatDesc* desc, size_t size = UINT_MAX);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                               hipArray_const_t array,
                                               const struct hipChannelFormatDesc& desc);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTextureToArray(struct texture<T, dim, readMode> *tex,
                                               hipArray_const_t array,
                                               const struct hipChannelFormatDesc* desc);

template <class T, int dim, enum hipTextureReadMode readMode>
inline static hipError_t hipBindTextureToArray(struct texture<T, dim, readMode>& tex,
                                               hipArray_const_t array);

template <class T>
inline static hipChannelFormatDesc hipCreateChannelDesc();

inline static hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w,
                                                        hipChannelFormatKind f);

inline static hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,
                                                const hipResourceDesc* pResDesc,
                                                const hipTextureDesc* pTexDesc,
                                                const hipResourceViewDesc* pResViewDesc);

inline static hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject);

inline static hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
                                                const hipResourceDesc* pResDesc);

inline static hipError_t hipDestroySurfaceObject(hipSurfaceObject_t surfaceObject);

inline static hipError_t hipGetTextureObjectResourceDesc(hipResourceDesc* pResDesc,
                                           hipTextureObject_t textureObject);

inline static hipError_t hipGetTextureAlignmentOffset(size_t* offset, const textureReference* texref);
inline static hipError_t hipGetChannelDesc(hipChannelFormatDesc* desc, hipArray_const_t array);

#define DUMMY_HIP_MAKE_VECTOR1(T, name) \
  struct name {\
    T x; \
  };


#define DUMMY_HIP_MAKE_VECTOR2(T, name) \
  struct name {\
    T x; \
    T y; \
  };


#define DUMMY_HIP_MAKE_VECTOR3(T, name) \
  struct name {\
    T x; \
    T y; \
    T z; \
  };


#define DUMMY_HIP_MAKE_VECTOR4(T, name) \
  struct name {\
    T x; \
    T y; \
    T z; \
    T w; \
  };

#define DUMMY_HIP_MAKE_VECTOR_TYPE(T, prefix) \
  DUMMY_HIP_MAKE_VECTOR1(T, prefix##1) \
  DUMMY_HIP_MAKE_VECTOR2(T, prefix##2) \
  DUMMY_HIP_MAKE_VECTOR3(T, prefix##3) \
  DUMMY_HIP_MAKE_VECTOR4(T, prefix##4)


DUMMY_HIP_MAKE_VECTOR_TYPE(signed char, char)
DUMMY_HIP_MAKE_VECTOR_TYPE(unsigned char, uchar)
DUMMY_HIP_MAKE_VECTOR_TYPE(short, short)
DUMMY_HIP_MAKE_VECTOR_TYPE(unsigned short, ushort)
DUMMY_HIP_MAKE_VECTOR_TYPE(int, int)
DUMMY_HIP_MAKE_VECTOR_TYPE(unsigned, uint)
DUMMY_HIP_MAKE_VECTOR_TYPE(long, long)
DUMMY_HIP_MAKE_VECTOR_TYPE(unsigned long, ulong)
DUMMY_HIP_MAKE_VECTOR_TYPE(long long, longlong)
DUMMY_HIP_MAKE_VECTOR_TYPE(unsigned long long, ulonglong)
DUMMY_HIP_MAKE_VECTOR_TYPE(float, float)
DUMMY_HIP_MAKE_VECTOR_TYPE(double, double)




float __fadd_rd(float x, float y);


float __fadd_rn(float x, float y);


float __fadd_ru(float x, float y);


float __fadd_rz(float x, float y);


float __fdiv_rd(float x, float y);


float __fdiv_rn(float x, float y);


float __fdiv_ru(float x, float y);


float __fdiv_rz(float x, float y);


float __fdividef(float x, float y);


float __fmaf_rd(float x, float y, float z);


float __fmaf_rn(float x, float y, float z);


float __fmaf_ru(float x, float y, float z);


float __fmaf_rz(float x, float y, float z);


float __fmul_rd(float x, float y);


float __fmul_rn(float x, float y);


float __fmul_ru(float x, float y);


float __fmul_rz(float x, float y);


float __frcp_rd(float x);


float __frcp_rn(float x);


float __frcp_ru(float x);


float __frcp_rz(float x);


float __frsqrt_rn(float x);


float __fsqrt_rd(float x);


float __fsqrt_rn(float x);


float __fsqrt_ru(float x);


float __fsqrt_rz(float x);


float __fsub_rd(float x, float y);


float __fsub_rn(float x, float y);


float __fsub_ru(float x, float y);


float __fsub_rz(float x, float y);





double __dadd_rd(double x, double y);


double __dadd_rn(double x, double y);


double __dadd_ru(double x, double y);


double __dadd_rz(double x, double y);


double __ddiv_rd(double x, double y);


double __ddiv_rn(double x, double y);


double __ddiv_ru(double x, double y);


double __ddiv_rz(double x, double y);


double __dmul_rd(double x, double y);


double __dmul_rn(double x, double y);


double __dmul_ru(double x, double y);


double __dmul_rz(double x, double y);


double __drcp_rd(double x);


double __drcp_rn(double x);


double __drcp_ru(double x);


double __drcp_rz(double x);


double __dsqrt_rd(double x);


double __dsqrt_rn(double x);


double __dsqrt_ru(double x);


double __dsqrt_rz(double x);


double __dsub_rd(double x, double y);


double __dsub_rn(double x, double y);


double __dsub_ru(double x, double y);


double __dsub_rz(double x, double y);


double __fma_rd(double x, double y, double z);


double __fma_rn(double x, double y, double z);


double __fma_ru(double x, double y, double z);


double __fma_rz(double x, double y, double z);


#endif // HIP_RUNTIME_H
