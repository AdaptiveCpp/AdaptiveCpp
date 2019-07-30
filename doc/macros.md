# Macros used by hipSYCL

# Read-only macros
Most of these are managed by `detail/backend/backend.hpp`.

* `__HIPSYCL__` - defined if compiling with hipSYCL
* `HIPSYCL_PLATFORM_CUDA` - defined if compiling for CUDA
* `HIPSYCL_PLATFORM_HCC` - defined if compiling for ROCm
* `HIPSYCL_PLATFORM_CPU` - defined if compiling purely for CPU
* `__HIPSYCL_DEVICE__` - defined if generating code for GPU
* `SYCL_DEVICE_ONLY` - defined if generating code for GPU
* `HIPSYCL_CLANG` - defined by `syclcc-clang` when compiling with the clang plugin

## Mainly for hipSYCL developers
* `__HIPSYCL_TRANSFORM__` defined by legacy `syclcc` during the source-to-source transformation step
* `HIPSYCL_UNIVERSAL_TARGET` - expands to `__host__ __device__`. Use for functions that should be available everywhere.
* `HIPSYCL_KERNEL_TARGET` - currently expands to `__host__ __device__`. Use for functions that should be available in kernels.
* `__HIPSYCL_DEVICE_CALLABLE__` - defined if HIP/CUDA intrinsics (e.g. `__syncthreads()`) are available. Since hipCPU defines those as well, this is also defined on CPU! However, it is _not_ defined when compiling for CUDA/ROCm and currently processing the `__host__` side.

# Configuration macros
* `HIPSYCL_EXT_FP_ATOMICS` - define before including `sycl.hpp` to enable the hipSYCL extension to allow atomic operations on floating point types. Since this is not in the spec, this may break portability. Additionally, not all hipSYCL backends may support the same set of FP atomics. It is the user's responsibility to ensure that the code remains portable and to implement fallbacks for platforms that don't support this.
