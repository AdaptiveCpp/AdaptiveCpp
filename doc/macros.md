# Macros used by AdaptiveCpp

# Read-only macros

## General macros
* `__ACPP__`, `__ADAPTIVECPP__` - defined if compiling with AdaptiveCpp

## Macros to specialize code paths based on backend

* `__hipsycl_if_target_host(code)` - `code` will only be compiled for the host backend.
* `__hipsycl_if_target_device(code)` - `code` will only be compiled for device backends.
* `__hipsycl_if_target_cuda(code)` - `code` will only be compiled for the CUDA backend.
* `__hipsycl_if_target_hip(code)` - `code` will only be compiled for the HIP backend.
* `__hipsycl_if_target_hiplike(code)` - `code` will only be compiled for the CUDA and HIP backend.

## Information about current compiler

### `HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_<backend>`

* `HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA` - Set to 1 if the compiler currently compiling the code supports CUDA language extensions. 0 otherwise.
* `HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP` - Set to 1 if the compiler currently compiling the code supports HIP language extensions. 0 otherwise.
* `HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HOST` - Always set to 1, since every compiler supports the host language, which is just C++.

## Information about compilation passes

### `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_<backend>`
Note: Some compiler drivers that AdaptiveCpp supports can compile for multiple backends in a single pass. Therefore, the following macros should not be seen as mutually exclusive in general. Currently, this affects the `cuda-nvcxx` driver which can target CUDA and host in a single pass.

* `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST` - Set to 1 if the current compilation pass targets host. 0 otherwise. 
* `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA` - Set to 1 if the current compilation pass targets CUDA. 0 otherwise. 
* `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP` - Set to 1 if the current compilation pass targets host. 0 otherwise. 
* `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV` - Set to 1 if the current compilation pass targets host. 0 otherwise. 

### Properties of current compilation pass

* `HIPSYCL_LIBKERNEL_IS_DEVICE_PASS` - Set to 1 if the current compilation pass targets at least one device backend. 0 otherwise.
* `HIPSYCL_LIBKERNEL_IS_EXCLUSIVE_PASS(backend)` - returns 1 if the current compilation pass targets the provided backend (`CUDA|HIP|HOST`) and no other backend.
* `HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS` - Set to 1 if the current compilation pass compiles for both host and device in a single, unified compilation pass.
* `SYCL_DEVICE_ONLY` - defined if the current compilation pass targets a device backend and `HIPSYCL_LIBKERNEL_IS_UNIFIED_HOST_DEVICE_PASS` is 0. **Note**: `SYCL_DEVICE_ONLY` is not defined for `cuda-nvcxx` where host and device are compiled in a single pass. This is therefore in general not suitable to implement specialized code paths for host and device in a portable way
* `__HIPSYCL_CLANG__` - defined by `acpp` when compiling with the clang plugin

## Information about targeted backends

* `__HIPSYCL_ENABLE_HIP_TARGET__` - defined during host and device passes if HIP is targeted
* `__HIPSYCL_ENABLE_CUDA_TARGET__` - defined during host and device passes if CUDA is targeted
* `__HIPSYCL_ENABLE_SPIRV_TARGET__` - defined during host and device passes if SPIR-V is targeted
* `__HIPSYCL_ENABLE_OMPHOST_TARGET__` - defined if OpenMP is targeted

## Extension feature test macros

* `HIPSYCL_EXT_<NAME>` - defined if the AdaptiveCpp extension `<NAME>` is available.

## Deprecated macros

* (deprecated) `HIPSYCL_PLATFORM_CUDA` - defined when CUDA language extensions are available
* (deprecated) `HIPSYCL_PLATFORM_ROCM`, `HIPSYCL_PLATFORM_HIP` - defined when HIP language extensions are available
* (deprecated) `HIPSYCL_PLATFORM_CPU` - defined if compiling for the host


## Mainly for AdaptiveCpp developers
* `HIPSYCL_UNIVERSAL_TARGET` - expands to `__host__ __device__`. Use for functions that should be available everywhere.
* `HIPSYCL_KERNEL_TARGET` - currently expands to `__host__ __device__`. Use for functions that should be available in kernels.

# Configuration macros
* `HIPSYCL_ENABLE_UNIQUE_NAME_MANGLING` - define during compilation of the AdaptiveCpp clang plugin to force enabling unique name mangling which is a requirement for explicit mulitpass compilation. This requires a clang that supports `__builting_unique_stable_name()`, and is automatically enabled on clang 11.
* `HIPSYCL_DEBUG_LEVEL` - sets the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is warning for Release and info for Debug builds.
* `HIPSYCL_STRICT_ACCESSOR_DEDUCTION` - define when building your SYCL implementation to enforce strict SYCL 2020 accessor type deduction rules. While this might be required for the correct compilation of certain SYCL code, it also disables parts of the AdaptiveCpp accessor variants performance optimization extension. As such, it can have a negative performance impact for code bound by register pressure.
* `HIPSYCL_ALLOW_INSTANT_SUBMISSION` - define to `1` before including `sycl.hpp` to allow submission of USM operations to in-order queues via the low-latency instant submission mechanism. Set to `0` to prevent the runtime from utilizing the instant submission mechanism. If C++ standard parallelism offloading is enabled, instant submissions are always allowed.

