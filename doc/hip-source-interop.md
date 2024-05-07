# Using platform-specific features in AdaptiveCpp (interoperability compilation flows only)

The interoperability-focused compilation flows `omp`, `cuda` and `hip` allow using backend-provided programming models like CUDA or HIP within SYCL kernels.

This can be used as follows:
```cpp
HIPSYCL_UNIVERSAL_TARGET
void optimized_codepaths()
{
  __hipsycl_if_target_cuda(
    // Only executed on CUDA device. CUDA specific device functions can be called here
  );
  __hipsycl_if_target_hip(
    // Only executed on HIP device. ROCm specific device functions can be called here
  );
  __hipsycl_if_target_host(
    // Host-specific code here. Since this runs exclusively on host, this can be any
    // arbitrary C++ code, and the usual SYCL kernel restrictions don't apply.
  );
}

...

q.parallel_for(range, [=](auto idx){
  optimized_codepaths();
});

```
Note that in general, CUDA or HIP `__device__` functions can only be called from functions that are marked as `__host__ __device__`, or the more portable `HIPSYCL_UNIVERSAL_TARGET`. The reason for this is that clang initially parses all SYCL code as host code, so only `__host__ __device__` functions can be called from kernels. Additionally, clang requires that host code must be present and correct even when compiling for device.