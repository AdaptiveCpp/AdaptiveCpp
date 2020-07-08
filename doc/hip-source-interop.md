# Using CUDA/HIP specific features in hipSYCL

Assume `kernel_function` is a function used in a SYCL kernel. Platform specific features can be used as follows:
```cpp
__host__ __device__
void optimized_function()
{
  // SYCL_DEVICE_ONLY checks if we are in the device compilation
  // pass
#ifdef SYCL_DEVICE_ONLY
  #ifdef HIPSYCL_PLATFORM_CUDA
  // CUDA specific device functions can be called here
  #elif defined(HIPSYCL_PLATFORM_ROCM)
  // ROCm specific device functions can be called here
  #endif
#endif
}

void kernel_function()
{
#if defined(__HIPSYCL__) && defined(SYCL_DEVICE_ONLY)
  optimized_function()
#else
  // Regular SYCL version here
#endif
}
```
This construct may seem slightly complicated. The reason for this is that clang initially parses all SYCL code as host code, so only `__host__ __device__` functions can be called from kernels. Additionally, clang requires that host code must be present and correct even when compiling for device.