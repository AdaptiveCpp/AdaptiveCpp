# Project history

* 2019/05/12: hipSYCL now uses a clang plugin to compile SYCL code *directly*, without source-to-source transformation
* 2019/01/30: hipSYCL now uses clang as default CUDA compiler instead of nvcc.
* 2019/01/18: an implementation of a CPU backend based on OpenMP has been merged into hipSYCL's master branch!
* 2018/12/24: hipSYCL is capable of compiling and running at least some SYCL parallel STL algorithms (more testing ongoing)
* 2018/10/21: The source-to-source transformation step can now also correctly treat header files; the recommended input language for hipSYCL is now regular SYCL.
* 2018/10/07: hipSYCL now comes with an experimental additional source-to-source transformation step during compilation which allows hipSYCL to ingest regular SYCL code.
* 2018/9/24: hipSYCL now compiles cleanly on ROCm as well.
* ... many more changes, including support for CPUs with LLVM compiler support, Level Zero runtime, generic single-pass compiler, ...
* 2023/2/9: hipSYCL was renamed to Open SYCL
* Sep 2023: Open SYCL was renamed to AdaptiveCpp due to external legal pressure and to account for a broadening of the project scope.
