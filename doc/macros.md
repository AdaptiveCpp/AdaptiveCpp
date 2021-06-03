# Macros used by hipSYCL

# Read-only macros

* `__HIPSYCL__` - defined if compiling with hipSYCL
* `HIPSYCL_PLATFORM_CUDA` - defined when CUDA language extensions are available
* `HIPSYCL_PLATFORM_ROCM`, `HIPSYCL_PLATFORM_HIP` - defined when HIP language extensions are available
* `HIPSYCL_PLATFORM_SPIRV` - defined if SPIR-V intrinsics are available 
* `HIPSYCL_PLATFORM_CPU` - defined if compiling for the host
* `SYCL_DEVICE_ONLY` - defined if generating code for GPU
* `__HIPSYCL_CLANG__` - defined by `syclcc-clang` when compiling with the clang plugin
* `HIPSYCL_EXT_<NAME>` - defined if the hipSYCL extension `<NAME>` is available.
* `__HIPSYCL_ENABLE_HIP_TARGET__` - defined during host and device passes if HIP is targeted
* `__HIPSYCL_ENABLE_CUDA_TARGET__` - defined during host and device passes if CUDA is targeted
* `__HIPSYCL_ENABLE_OMPHOST_TARGET__` - defined if OpenMP is targeted

## Mainly for hipSYCL developers
* `HIPSYCL_UNIVERSAL_TARGET` - expands to `__host__ __device__`. Use for functions that should be available everywhere.
* `HIPSYCL_KERNEL_TARGET` - currently expands to `__host__ __device__`. Use for functions that should be available in kernels.

# Configuration macros
* `HIPSYCL_ENABLE_UNIQUE_NAME_MANGLING` - define during compilation of the hipSYCL clang plugin to force enabling unique name mangling which is a requirement for explicit mulitpass compilation. This requires a clang that supports `__builting_unique_stable_name()`, and is automatically enabled on clang 11.
* `HIPSYCL_DEBUG_LEVEL` - sets the output verbosity. `0`: none, `1`: error, `2`: warning, `3`: info, `4`: verbose, default is warning for Release and info for Debug builds.
* `HIPSYCL_STRICT_ACCESSOR_DEDUCTION` - define when building your SYCL implementation to enforce strict SYCL 2020 accessor type deduction rules. While this might be required for the correct compilation of certain SYCL code, it also disables parts of the hipSYCL accessor variants performance optimization extension. As such, it can have a negative performance impact for code bound by register pressure.
