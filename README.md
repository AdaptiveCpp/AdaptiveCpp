# hipSYCL - an implementation of SYCL over NVIDIA CUDA/AMD HIP
The goal of the hipSYCL project is to develop a SYCL 1.2.1 implementation that is built upon NVIDIA CUDA/AMD HIP.

hipSYCL provides a SYCL interface to NVIDIA CUDA and AMD HIP. hipSYCL applications are then compiled with the regular vendor compilers (nvcc for nvidia and hcc for AMD), and hence can enjoy full vendor support.

## Why use hipSYCL over raw CUDA/HIP?
* hipSYCL provides a modern C++ API, including automatic resource management via reference counting semantics (see the SYCL spec for more details). No more worrying about missing cudaFree() calls.
* Openness. hipSYCL applications are written against an API that is an open standard (SYCL) instead of being locked to one specific vendor.
* Portability. The hipSYCL input language is (at the moment) a slightly modified, annotated SYCL (see the corresponding section below). This input language can be turned into regular SYCL with simple `#defines`. A hipSYCL application can hence run
   * via hipSYCL:
      * on AMD devices via AMD HIP on the ROCm platform
      * on NVIDIA devices via CUDA
   * via regular SYCL:
      * the triSYCL SYCL implementation can be used to execute the application
         * on CPUs with OpenMP
         * on Xilinx FPGAs (experimental)
      * Codeplay's ComputeCpp SYCL implementation runs on any OpenCL device with SPIR support, including at least:
         * on AMD devices
         * on Intel devices
* Powerful, but intuitive programming model. SYCL (and by extension, hipSYCL) relies on an asynchronous task graph with support for out-of-order execution instead of the simple in-order queues (streams) provided by CUDA and HIP. This task graph is constructed based on the requirements (e.g. memory accesses) that the user specifies for one kernel. All data transfers between host and device are then executed automatically (if necessary) by the SYCL runtime. hipSYCL will optimize the execution order of the tasks and will for example automatically try to overlap kernels and data transfers, if possible. This allows for the development of highly optimized applications with little effort from the application developer.
* All CUDA or HIP intrinsics or other features can still be used from within hipSYCL if desired. This is because an hipSYCL application is compiled like any regular CUDA/HIP application with nvcc/hcc. For portability, it is however best to use such unportable features only within preprocessor `#ifdef HIPSYCL_PLATFORM_CUDA` or `#ifdef HIPSYCL_PLATFORM_HIP` constructs.

## Current state
hipSYCL is still in an experimental stage of development. It can successfully execute some SYCL programs; but parts of the specification are not yet implemented.

Still unimplemented/missing is in particular:
* hierarchical kernel dispatch with flexible work group ranges (hierarchical dispatch with ranges fixed at `parallel_for_work_group` invocation is supported).
* Explicit memory copy functions
* placeholder accessors
* Atomics
* Device library
* Images
* queue and event information queries
* vec<> class is incomplete
* Some locks for multithreaded SYCL applications are missing.
* Everything related to SYCL's OpenCL interoperability features. This is because hipSYCL uses HIP/CUDA as backend instead of OpenCL.

## Building hipSYCL
In order to successfully build and install hipSYCL, a working installation of either CUDA or ROCm (with nvcc/hcc in `$PATH`) is required. At the moment, hipSYCL is tested with CUDA 9.2 and gcc 7.3.

For Arch Linux users, it is recommended to simply use the `PKGBUILD` provided in `install/archlinux`. A simple `makepkg` in this directory should be enough to build an Arch Linux package.

All other users need to compile hipSYCL manually. First, make sure to clone the repository with all submodules:
```
$ git clone --recurse-submodules https://github.com/illuhad/hipSYCL
```
Then, create a build directory and compile hipSYCL:
```
$ cd <build directory>
$ cmake -DCMAKE_INSTALL_PREFIX=<installation prefix> <hipSYCL source directory>
$ make install
```
The default installation prefix is `/usr/local`. Change this to your liking.

## hipSYCL input language: `__device__` annotated SYCL
Due to the CUDA/HIP programming model that hipSYCL builds upon, all functions that are executed on the device must be annotated with the `__device__` attribute. This particularly affects SYCL kernel lambdas. Local memory allocated statically in an hierarchical parallel for invocation must be marked as `__shared__` for the same reason. The following code snippet can be used to guarantee compatibility of hipSYCL applications with regular SYCL:
```cpp
#ifndef __HIPSYCL__
 #define __device__
 #define __host__
 #define __global__
 #define __shared__
#endif
```
As a future project, it is planned to investigate the possibility of automatically adding these annotations with libclang, such that unmodified, regular SYCL code can be compiled as well.

## Caveats
* hipSYCL uses a slightly different input language compared to regular SYCL, see above
* hipSYCL uses AMD HIP as backend, which in turn can target CUDA and AMD devices. Due to lack of hardware, unfortunately hipSYCL is untested on AMD at the moment. Bug reports (or better, reports of successes) are greatly appreciated.
* Because hipSYCL isn't based on OpenCL, all SYCL OpenCL interoperability features will very likely never be available in hipSYCL.

## Compiling software with hipSYCL
hipSYCL provides the `syclcc` compiler wrapper. `syclcc` will automatically call either nvcc or hcc, depending on what is installed. If both are installed, the `HIPSYCL_PLATFORM` environment variable can be used to select the compiler (set to "cuda" or "nvcc" for nvidia, and "hip", "rocm" or "hcc" for AMD). `syclcc` also automatically sets a couple of compiler flags required for the compilation of hipSYCL programs. All other arguments are forwarded to hcc/nvcc.

## Example
The following code adds two vectors:
```cpp
#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

#ifndef __HIPSYCL__
// This guarantees compatibility if we are not using hipSYCL as SYCL implementation.
#define __device__
#endif

using data_type = float;

std::vector<data_type> add(cl::sycl::queue& q,
                           const std::vector<data_type>& a,
                           const std::vector<data_type>& b)
{
  std::vector<data_type> c(a.size());

  assert(a.size() == b.size());
  cl::sycl::range<1> work_items{a.size()};

  {
    cl::sycl::buffer<data_type> buff_a(a.data(), a.size());
    cl::sycl::buffer<data_type> buff_b(b.data(), b.size());
    cl::sycl::buffer<data_type> buff_c(c.data(), c.size());

    q.submit([&](cl::sycl::handler& cgh){
      auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_c = buff_c.get_access<cl::sycl::access::mode::write>(cgh);

      cgh.parallel_for<class vector_add>(work_items,
                                         [=] __device__ (cl::sycl::id<1> tid) {
        access_c[tid] = access_a[tid] + access_b[tid];
      });
    });
  }
  return c;
}

int main()
{
  cl::sycl::queue q;
  std::vector<data_type> a = {1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<data_type> b = {-1.f, 2.f, -3.f, 4.f, -5.f};
  auto result = add(q, a, b);

  std::cout << "Result: " << std::endl;
  for(const auto x: result)
    std::cout << x << std::endl;
}

```



