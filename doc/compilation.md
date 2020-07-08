# hipSYCL compilation model

hipSYCL relies on the fact that both HIP/CUDA and SYCL are single-source programming models based on C++. In principle, this allows SYCL to be implemented as a HIP/CUDA library that simply wraps CUDA or HIP device or runtime functions. This wrapper could be very thin in practice with negligible performance impact due to aggressive inlining by device compilers. The SYCL application could then be compiled with regular CUDA or ROCm compilers. As a side effect, this approach also brings access to CUDA and HIP specific features from SYCL code. This is exactly the idea behind hipSYCL.

Reality is unfortunately more complicated because the HIP/CUDA programming model is more restrictive than SYCL. For example, CUDA and HIP require that functions should be explicitly marked by the programmer whether they should be compiled for device or host. SYCL does not require this. To account for such restrictions, hipSYCL uses a clang plugin that adds additional AST and IR passes to augment clang's CUDA/HIP support to also support SYCL code.
This process is fully integrated in the `syclcc-clang` compiler wrapper, so users can just call `syclcc-clang` like a regular compiler and do not need to worry about the details. Users can also use `syclcc` (an alias for `syclcc-clang`) if they wish to save a couple of letters when typing.

See below for an illustration of the hipSYCL compilation model.

<img src="img/hipsycl-compilation.png" width="300">