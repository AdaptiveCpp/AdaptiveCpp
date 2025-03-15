# AdaptiveCpp portable CUDA (PCUDA)

**Note**: PCUDA is under development and in WIP/experimental state. Only a small subset of the target functionality is currently implemented (see feature support further down)

AdaptiveCpp supports a dialect of the CUDA/HIP language called portable CUDA (PCUDA). This is exclusively supported with the generic JIT compiler (`--acpp-targets=generic`), and is supported on all backends.
Similarly to using `--acpp-targets=generic` with SYCL or C++ standard parallelism code (stdpar), AdaptiveCpp can JIT-compile the input PCUDA code for any target from a single binary, including CPUs, NVIDIA GPUs, AMD GPUs, and Intel GPUs.

Possible use cases for PCUDA include:
* Working with existing CUDA/HIP code bases;
* Iterative porting of existing CUDA/HIP code bases;
* Fairer compiler comparisons between AdaptiveCpp and nvcc or hipcc, since the exact same (or at least very similar) input code can be used for benchmarks;
* Preference for a lower-level, more C-style API compared to what SYCL or stdpar offers;
* When a more stable ABI for the runtime library is needed, since PCUDA has a C instead of a C++ interface like SYCL;
* Significantly lower compile times and lower kernel submission latency compared to SYCL.

While PCUDA shares the same lower layers in our compiler and runtime as SYCL or stdpar, it is *not* implemented on top of SYCL. Instead, it directly ties into those lower layers, in a way that is designed to be interoperable with SYCL. This allows PCUDA compile times to be drastically lower compared to SYCL, where compile times are often heavily affected by the large `sycl.hpp` C++ header.

PCUDA is not intended as a replacement for SYCL; it is a tool with a different profile for different use cases and preferences. Users with a preference for modern C++ or abstractions should use SYCL or stdpar.

**Note:** AdaptiveCpp PCUDA is not a tool to enable running proprietary NVIDIA or AMD code on other hardware. We do not support any use case that breaks the CUDA EULA. AdaptiveCpp does not recompile assembly or binaries. Using AdaptiveCpp to compile vendor libraries available in source form is not supported and unlikely to work due to fundamental compiler differences -- see the section on differences between PCUDA and HIP/CUDA below!

## Enabling PCUDA

PCUDA mode is activated using `--acpp-pcuda`.

If you need triple chevron kernel launch syntax (i.e. `kernel<<<>>>>()` syntax, not recommended for developing production code, see the section on kernel launch below), use `--acpp-pcuda-chevron-launch`.

## Differences between PCUDA and HIP/CUDA

AdaptiveCpp's generic JIT compiler is a single-pass JIT compiler, and as such substantially different from either nvcc, hipcc, clang CUDA or nvc++.

The AdaptiveCpp compiler design allows for lower compile times and and better portability and performance due to JIT, but certain features may work differently as a consequence.

### Code specialization for host/device


Due to the single-pass nature of AdaptiveCpp's compiler, host and device code are parsed together by a unified compiler, and there are consequently no macros to distinguish host and device code. Instead, `if(__acpp_sscp_is_device) {...}` and `if(__acpp_sscp_is_host){...}` can be used. These branches are evaluated by  the compiler and LLVM optimizer, and do not introduce overhead.

`__CUDA_ARCH__` is not defined!

### No compile-time knowledge of targets

Due to the JIT nature, the target device is not known at compile-time. Therefore, it is not possible to specialize code paths for different devices using preprocesor macros.
Instead, control flow and AdaptiveCpp's JIT reflection and `compile_if` functionality must be used (**Note**: This API is currently available in SYCL and can be used from PCUDA, however, a native PCUDA version will follow in the future).

Another consequence of the JIT nature is that `warpSize` is not `constexpr`, since the target device and its warp size is not known at compile time.

### Default-stream semantics

AdaptiveCpp does not implement CUDA's legacy default stream semantics and only supports CUDA's newer per-thread default stream semantics. ([More information](https://developer.nvidia.com/blog/gpu-pro-tip-cuda-7-streams-simplify-concurrency/)).

### Multi-backend device topology and device management

AdaptiveCpp is a portable platform supporting multiple backends that might potentially also be used simultaneously.
* Multiple backends may be available. The OpenMP host backend targeting the CPU is always available.
* A backend may contain multiple platorms. This is primarily used on backends like OpenCL, where the backend itself might expose multiple available drivers. For example, a system might have the Intel OpenCL implementation for CPU installed as well as the Intel GPU OpenCL implementation.
* A platform may contain multiple devices.

Because devices from different vendors in general do not know how to talk to each other, in general data transfers are only possible within devices of one platform (and the host).

Consequently, in-order to not break existing CUDA/HIP applications that rely on `cuda/hipGetDevice` and `cuda/hipSetDevice` and assume that all devices accessible in this way are interoperable, AdaptiveCpp PCUDA enumerates every platform and backend independently.

This means that the `GetDevice`/`SetDevice` functions always operate on devices of the current platform.

For example, an enumeration might look as follows:
```
Backend 0 (OpenMP host):
  Platform 0:
    Device 0: Host CPU
Backend 1 (CUDA):
  Platform 0:
    Device 0: NVIDIA RTX 4090 GPU #0
    Device 1: NVIDIA RTX 4090 GPU #1
Backend 2 (HIP):
  Platform 0: 
    (no devices)
Backend 3 (OpenCL):
  Platform 0 (Intel CPU OpenCL):
    Device 0: Host CPU
  Platform 1 (Intel GPU OpenCL):
    Device 0: Integrated Graphics

```

Utilizing devices from other platforms or backends requires the PCUDA extensions `pcudaGet/SetPlatform` and `pcudaGet/SetBackend` ([details](#managing-backends-and-platforms)).


### __host__, __device__ and __global__ attributes

As in SYCL or stdpar, AdaptiveCpp mostly figures out by itself which functions need to be compiled for device and which for host.
The `__host__` attribute is available, but meaningless in PCUDA. The `__device__` attribute is only needed on device functions that are invoked from other translation units.

`__host__` and `__device__` functions can otherwise call each other arbitrarily. (This is a necessary requirement for interoperability with SYCL device code, where functions don't have any attributes at all, and are therefore "host" in CUDA logic).

Unlike hipcc or clang CUDA, but similarly to nvcc, functions in PCUDA **cannot** be overloaded based on whether they have the `__host__` or `__device__` attribute:

```c++
// This code works in clang CUDA and hipcc, but not in nvcc and AdaptiveCpp PCUDA.
__host__ void myfunction() {
  // host implementation
}
__device__ void myfunction() {
  // device implementation
}

```

Functions with `__global__` attribute are compiled as kernels. Unlike other CUDA/HIP dialects, PCUDA allows kernels to also be invoked from device code defined in the same translation unit with a regular function call (i.e. not using a kernel launch mechanism). This then corresponds to a normal function call:

```c++

__global__ void copy(int* in, int* out, int num_elements) {...}

__global__ void kernel(int* in, int* out, int num_elements) {
  // This not dynamic parallelism; the semantics are the same as if
  // `copy` did not have the `__global__` attribute!
  copy(in, out, num_elements);
}

```

## Headers

Depending on your use case, you may wish to include different headers:

* `<pcuda.hpp>`: The main PCUDA header, recommended for PCUDA-exclusive code. Exposes the API with `pcuda` prefix, e.g. `pcudaMalloc()`.
* `<cuda_runtime_api.h>`, `<cuda_runtime.h>`, `<cuda.h>`: For compatibility with CUDA code. Exposes the  API with `cuda` prefix, e.g. `cudaMalloc()`.
* `<hip/hip_runtime.h>`, `<hip/hip_runtime_api.h>`. For compatibility with HIP code. Exposes the runtime API with `hip` prefix, e.g. `hipMalloc()`.

## Kernel launch

AdaptiveCpp PCUDA supports multiple kernel launch mechanisms:

### pcudaParallelFor

`pcudaParallelFor` is the native AdaptiveCpp PCUDA kernel launch mechanism. We recommend that production code use this mechanism. It is conceptually similar to e.g. a SYCL kernel launch, and can in principle support the same rich functionality as a SYCL kernel launch.

This is a PCUDA extension, and only available with `pcuda` prefix.

```c++
template <class F>
pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t shared_mem,
                             pcudaStream_t stream, F f);
template<class F>
pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t shared_mem, F f);

template<class F>
pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, F f);
```

`f` is a callable with no arguments that will be invoked in the generated kernel.

Example:

```c++

float* in = ...
float* out = ...
// launch 4 blocks with 256 threads
pcudaParallelFor(4, 256, [=](){
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  out[gid] = in[gid];
});
```

Note that (like in SYCL) every declaration of a new lambda passed into `pcudaParallelFor` causes a new kernel instantiation. If you want to invoke the same kernel from multiple places in your program, to avoid unnecessary JIT compilation of multiple kernels, better declare the lambda once and reuse it, or declare a function object for your kernel.

### pcudaLaunchKernelGGL

AdaptiveCpp PCUDA supports `pcudaLaunchKernelGGL` (including with either `hip` or `cuda` prefixes) in analogy to AMD's `hipLaunchKernelGGL`.

Note that this kernel invocation mode currently requires that the kernel is invoked from the same translation unit where the `__global__` function is defined.

Example:

```c++
__global__ void copy(float* in, float* out) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  out[gid] = in[gid];
}

void invoke_kernel() {
  float* in = ...
  float* out = ...
  // launch 4 blocks with 256 threads
  pcudaLaunchKernelGGL(copy, 4, 256, 0 /* shared memory */, 0 /* default stream*/, in, out);
}
```
### Chevron launch (`<<<>>>` syntax)

AdaptiveCpp supports the traditional CUDA triple chevron kernel launch syntax only for convenience and compatibility reasons on a best effort basis. It needs to be explicitly activated using `--acpp-pcuda-chevron-launch`. It is discouraged for new code investments.

This feature is currently implemented using a preprocessing step. While the expectation is that the implementation works well for actual CUDA/HIP code, the additional preprocessing step can affect compiler diagnostics for macro expansions negatively. This mode is therefore not recommended for code bases under active development.

Note that this kernel invocation mode currently requires that the kernel is invoked from the same translation unit where the `__global__` function is defined.

Example:
```c++
__global__ void copy(float* in, float* out) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  out[gid] = in[gid];
}

void invoke_kernel() {
  float* in = ...
  float* out = ...
  // launch 4 blocks with 256 threads
  copy<<<4, 256>>>(in, out);
}
```


## Device library

Supported features in device code include:

| Feature | Notes |
|------------------|-------------------|
|`threadIdx.x`| Also `hipThreadIdx_x` with the HIP header |
|`threadIdx.y`| Also `hipThreadIdx_y` with the HIP header |
|`threadIdx.z`| Also `hipThreadIdx_z` with the HIP header |
|`blockIdx.x`| Also `hipBlockIdx_x` with the HIP header |
|`blockIdx.y`| Also `hipBlockIdx_y` with the HIP header |
|`blockIdx.z`| Also `hipBlockIdx_z` with the HIP header |
|`blockDim.x`| Also `hipBlockDim_x` with the HIP header |
|`blockDim.y`| Also `hipBlockDim_y` with the HIP header |
|`blockDim.z`| Also `hipBlockDim_z` with the HIP header |
|`gridDim.x`| Also `hipGridDim_x` with the HIP header |
|`gridDim.y`| Also `hipGridDim_y` with the HIP header |
|`gridDim.z`| Also `hipGridDim_z` with the HIP header |
|Static local memory | E.g. `__shared__ int x;` syntax |
|Dynamic local memory | E.g. `extern __shared__ int x;` syntax |
|`warpSize` | Not `constexpr` in PCUDA! |
|`__syncthreads` | |
|`__syncwarp` | |
| Math functions from `<math.h>`/`<cmath>` | |
| CUDA/HIP vector types | |

## Supported runtime APIs

Unless otherwise noted, the functions listed below are also available with `hip`/`cuda` prefixes, depending on which header is included. (see the section on [headers](#headers) for details).

Refer to the CUDA or HIP documentation for more information on these functions.

| Feature | Notes |
|------------------|-------------------|
|`pcudaGetDeviceCount`| |
|`pcudaGetDevice`| |
|`pcudaSetDevice`| |
|`pcudaGetLastError` | |
|`pcudaPeekAtLastError` | |
|`pcudaGetErrorName` | |
|`pcudaGetErrorString` | |
|`pcudaDeviceSynchronize` | |
|`pcudaThreadSynchronize` | |
|`pcudaMalloc` | |
|`pcudaMallocHost` | |
|`pcudaHostAlloc` | Flags are ignored; behaves like pcudaMallocHost |
|`pcudaMallocManaged` | optional `flags` argument is currently ignored. |
|`pcudaFree` | |
|`pcudaFreeHost` | |
|`pcudaStreamCreate` | |
|`pcudaStreamCreateWithFlags` | flags are currently ignored |
|`pcudaStreamCreateWithPriority` | flags are currently ignored |
|`pcudaStreamDestory` | |
|`pcudaStreamSynchronize` | |
|`pcudaStreamWaitEvent` | flags are currently ignored |
|`pcudaMemcpy` | copy direction is ignored |
|`pcudaMemcpyAsync` | copy direction is ignored |
|`pcudaMemset` | |
|`pcudaMemsetAsync` | |
|`pcudaEventCreate` | |
|`pcudaEventCreateWithFlags` | Flags are currently ignored |
|`pcudaEventDestroy` | |
|`pcudaEventQuery` | |
|`pcudaEventRecord` | |
|`pcudaEventRecordWithFlags` | Flags are currently ignored |
|`pcudaEventSynchronize` | |
|`pcudaGetDeviceProperties` | |
|`pcudaDriverGetVersion` | Currently always returns 0 |

### Extensions


Functions that are PCUDA extensions and thus PCUDA-specific are only available with `pcuda` prefix.

#### Managing backends and platforms

PCUDA relies on a hierarchical device topology where multiple backends may be available, each consisting potentially of multiple platforms, each of which may support multiple devices.
Only devices of the same platform are interoperable (e.g. can transfer memory between each other using `pcudaMemcpy`).
Switching between platforms and backends can be achieved with these PCUDA extensions:

```c++

/// Returns the number of platforms within the current backend
ACPP_PCUDA_API pcudaError_t pcudaGetPlatformCount(int* count);
/// Returns the index of the current platform within the current backend
ACPP_PCUDA_API pcudaError_t pcudaGetPlatform(int* platform);
/// Sets the index of the current platform within the current backend.
/// Also resets the active device similarly to
/// pcudaSetDevice(0).
ACPP_PCUDA_API pcudaError_t pcudaSetPlatform(int platform);

/// Returns the number of available backends
ACPP_PCUDA_API pcudaError_t pcudaGetBackendCount(int* count);
/// Returns the index of current backend
ACPP_PCUDA_API pcudaError_t pcudaGetBackend(int* backend);
/// Sets the currently active backend
/// Also resets the active device and platform similarly to
/// pcudaSetPlatform(0).
ACPP_PCUDA_API pcudaError_t pcudaSetBackend(int backend);

/// Sets backend, platform and device at the same time.
ACPP_PCUDA_API pcudaError_t pcudaSetDeviceExt(int backend, int platform, int device);

```

They return `pcudaSuccess` in case of success, and `pcudaErrorInvalidValue` in case of error (e.g. the pointers are null, or the enumeration indices are invalid).
In case the backend or platform is changed to a backend or platform without devices, `pcudaErrorNoDevice` is returned.

Note that in general, the order in which `pcudaSet*` calls on different levels of the hierarchy are made is important: For example, `pcudaSetDevice` only considers devices from the current platform. If the device index is invalid in the current platform, `pcudaSetDevice()` may fail, even if afterwards a `pcudaSetPlatform()` is used to set the platform to one where the device index may be valid.
Therefore, the only correct order when multiple levels of the hierarchy need to be changed is 

```c++
pcudaSetBackend(backend);
pcudaSetPlatform(platform);
pcudaSetDevice(device);
```

`pcudaSetDeviceExt()` takes this into account and may thus be more convenient and safer.

## Interoperability with SYCL or C++ standard parallelism in device code

When using `--acpp-targets=generic`, all programming models supported by AdaptiveCpp are interchangable arbitrarily in device code.
You can mix-and-match code as needed with any of the programming models.

Example:

```c++

sycl::queue q;
float* data = ...
q.parallel_for(range, [=](auto idx){
  // Note: Include <pcuda.hpp> and compile with --acpp-pcuda
  int gid = threadIdx.x + blockIdx.x * blockDim.x;
  data[gid] *= 2.5f;
});

```

## Interoperability with SYCL for runtime objects

TBD

