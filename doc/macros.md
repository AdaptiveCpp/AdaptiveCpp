# Macros used by hipSYCL

# Read-only macros
Most of these are managed by `detail/backend/backend.hpp`.

* `__HIPSYCL__` - defined if compiling with hipSYCL
* `HIPSYCL_PLATFORM_CUDA` - defined if compiling for CUDA
* `HIPSYCL_PLATFORM_HCC` (**deprecated**) - defined if compiling for ROCm
* `HIPSYCL_PLATFORM_ROCM` - defined if compiling for ROCm
* `HIPSYCL_PLATFORM_CPU` - defined if compiling purely for CPU
* `__HIPSYCL_DEVICE__` - defined if generating code for GPU
* `SYCL_DEVICE_ONLY` - defined if generating code for GPU
* `HIPSYCL_CLANG` - defined by `syclcc-clang` when compiling with the clang plugin
* `HIPSYCL_SVM_SUPPORTED` - defined when the backend supports unified memory
* `HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE` - defined if hipSYCL supports (and has enabled) the extension for automatically requiring placeholder accessors.
* `HIPSYCL_EXT_CUSTOM_PFWI_SYNCHRONIZATION` - defined if the hipSYCL custom `parallel_for_work_item` synchronization extension is supported.

## Mainly for hipSYCL developers
* `HIPSYCL_UNIVERSAL_TARGET` - expands to `__host__ __device__`. Use for functions that should be available everywhere.
* `HIPSYCL_KERNEL_TARGET` - currently expands to `__host__ __device__`. Use for functions that should be available in kernels.
* `__HIPSYCL_DEVICE_CALLABLE__` - defined if HIP/CUDA intrinsics (e.g. `__syncthreads()`) are available. Since hipCPU defines those as well, this is also defined on CPU! However, it is _not_ defined when compiling for CUDA/ROCm and currently processing the `__host__` side.

# Configuration macros
* `HIPSYCL_EXT_ENABLE_ALL` - define before including `sycl.hpp` to enable all extensions.
* `HIPSYCL_EXT_FP_ATOMICS` - define before including `sycl.hpp` to enable the hipSYCL extension to allow atomic operations on floating point types. Since this is not in the spec, this may break portability. Additionally, not all hipSYCL backends may support the same set of FP atomics. It is the user's responsibility to ensure that the code remains portable and to implement fallbacks for platforms that don't support this.
* `HIPSYCL_CPU_EMULATE_SEPARATE_MEMORY` - define during compilation of hipSYCL and SYCL programs to force the CPU backend to consider host and device memory as separate memory regions that require data transfers in between. This can be useful for debugging since it recreates what happens when targeting GPUs.
* `HIPSYCL_USE_KERNEL_NAMES` - define during compilation of the hipSYCL clang plugin to control whether hipSYCL should use the kernel names provided to invocation functions such as `parallel_for<KernelName>`. If enabled (currently the default), each kernel invocation has to be provided a unique kernel name. While hipSYCL technically doesn't require this, it is currently recommended to enable this to work around a clang bug (see #49).

