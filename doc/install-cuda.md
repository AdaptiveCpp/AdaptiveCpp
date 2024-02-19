# AdaptiveCpp installation instructions for CUDA

## If using clang

Please install CUDA 10.0 or later.

clang usually produces CUDA programs with very competitive performance compared to nvcc or nvc++. For more information on compiling CUDA with clang, please read [the LLVM documentation on CUDA support](http://llvm.org/docs/CompileCudaWithLLVM.html). **Note that the requirements on the CUDA installation described there.**

If you use a very recent CUDA version, you might get a warning when compiling with AdaptiveCpp that clang does not support your CUDA version and treats like an older version. This warning can usually safely be ignored.

CMake variables:
* `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda` to point AdaptiveCpp to the CUDA root installation directory (e.g. `/usr/local/cuda`), if cmake doesn't find the right CUDA installation.
* `-DWITH_CUDA_BACKEND=ON` if AdaptiveCpp does not automatically enable the CUDA backend 

## If using nvc++

Please install the latest release of the NVIDIA HPC SDK and make sure to point AdaptiveCpp to nvc++ (see below).
Please install CUDA 10.0 or later. You can also rely on the CUDA bundled with the NVIDIA HPC SDK

CMake variables:
* `-DNVCXX_COMPILER=/path/to/nvc++`
* You can use the CUDA bundled with nvc++. Make sure to point AdaptiveCpp to the right CUDA installation using `-DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda`. 
* `-DWITH_CUDA_BACKEND=ON` if AdaptiveCpp does not automatically enable the CUDA backend
* `-DWITH_CUDA_NVCXX_ONLY=ON` enable if you want to use the CUDA backend exclusively with nvc++ and not clang. This will allow you to use nvc++ without having to install LLVM.