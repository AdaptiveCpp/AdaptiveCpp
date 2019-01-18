# hipSYCL - an implementation of SYCL over NVIDIA CUDA/AMD HIP
[![Build Status](https://travis-ci.com/illuhad/hipSYCL.svg?branch=master)](https://travis-ci.com/illuhad/hipSYCL)

The goal of the hipSYCL project is to develop a SYCL 1.2.1 implementation that builds upon NVIDIA CUDA/AMD HIP. hipSYCL consists of a source-to-source transformation toolchain to automatically convert SYCL code into CUDA/HIP code behind the scenes, and an implementation of a SYCL runtime that runs on top of CUDA/HIP. The actual compilation step is then carried out with the regular NVIDIA and AMD compilers nvcc/hcc, which allows hipSYCL applications to make use of all the latest nvcc/hcc features (e.g. intrinsics). This approach also guarantees compatibility of hipSYCL applications with established and well-supported vendor tools (e.g. profilers) and libraries in the NVIDIA and AMD ecosystems.

## Why use hipSYCL over raw CUDA/HIP?
* hipSYCL provides a modern C++ API, including automatic resource management via reference counting semantics (see the SYCL spec for more details). No more worrying about missing cudaFree() calls. Unlike CUDA/HIP, (hip)SYCL does not require explicitly marking functions as `__host__` or `__device__` - the SYCL compiler will figure that out on its own.
* Openness. hipSYCL applications are written against an API that is an open standard (SYCL) instead of being locked to one specific vendor.
* Portability. hipSYCL ingests regular SYCL code which can also be executed on a variety of other SYCL implementations targeting different hardware. This is illustrated in the following image:
  ![SYCL implementations](/doc/img/sycl-targets.png)
* Powerful, but intuitive programming model. SYCL (and by extension, hipSYCL) relies on an asynchronous task graph with support for out-of-order execution instead of the simple in-order queues (streams) provided by CUDA and HIP. This task graph is constructed based on the requirements (e.g. memory accesses) that the user specifies for one kernel. All data transfers between host and device are then executed automatically (if necessary) by the SYCL runtime. hipSYCL will optimize the execution order of the tasks and will for example automatically try to overlap kernels and data transfers, if possible. This allows for the development of highly optimized applications with little effort from the application developer.
* All CUDA or HIP intrinsics or other features can still be used from within hipSYCL if desired. This is because an hipSYCL application is compiled like any regular CUDA/HIP application with nvcc/hcc. Since hipSYCL attempts to parse the input source as regular SYCL, you _must_ surround the use of CUDA or HIP specific features with `#ifdef HIPSYCL_PLATFORM_CUDA` or `#ifdef HIPSYCL_PLATFORM_HCC` constructs.

## How it works
hipSYCL relies on the fact that that both HIP/CUDA and SYCL are single-source programming models based on C++. In principle, this allows SYCL to be implemented as a HIP/CUDA library that simply wraps CUDA or HIP device or runtime functions (This wrapper could be very thin in practice with negligible performance impact due to aggressive inlining by nvcc and hcc). The SYCL application can then be compiled with the regular NVIDA or AMD compilers nvcc/hcc. As a side effect, this approach also brings access to CUDA and HIP specific features from SYCL code. This is exactly the idea behind hipSYCL.

In reality, it is more complicated though because the HIP/CUDA programming model is more restrictive than SYCL. For example, CUDA and HIP require that functions should be explicitly marked by the programmer whether they should be compiled for device or host. SYCL doesn't require this. To fix such restrictions, hipSYCL performs a source-to-source transformation of the input source before it is fed into nvcc or hcc. In particular, during this step `__shared__`, `__host__` and `__device__` attributes are automatically inserted where needed. These steps are executed transparently for the user by hipSYCL's `syclcc` compilation driver. See below for an illustration of the individual steps for the compilation targets of hipSYCL.
![hipSYCL compilation](/doc/img/hipsycl-compilation-small.png)

## Current state
hipSYCL is still in an early stage of development. It can successfully execute many SYCL programs; but parts of the specification are not yet implemented.

### Limitations
#### Unimplemented major features 
* hierarchical kernel dispatch with flexible work group ranges (hierarchical dispatch with ranges fixed at `parallel_for_work_group` invocation is supported).
* Explicit memory copy functions (partially implemented)
* Atomics
* Device builtins (math library is mostly complete though)
* Images
* vec<> class lacks convert(), as(), swizzled temporary vector objects lack operators
* Some locks for multithreaded SYCL applications are missing.
* Error handling: wait_and_throw() and throw_asynchronous() do not invoke async handler
* 0-dimensional objects (e.g 0D accessors) are mostly unimplemented
* Because hipSYCL isn't based on OpenCL, all SYCL OpenCL interoperability features are unimplemented and may very likely never be available in hipSYCL.

#### Other limitations
* On NVIDIA: In some cases, named kernels hit restrictions of NVIDIA's implementation of extended lambdas (lambdas that are used to instantiate templated kernels). NVIDIA does not allow kernel names to be local types in general. This especially affects names of execution policies in the SYCL parallel STL library. Workaround: Use unnamed kernels (or execution policies) or use global types. See [NVIDIA's official documentation of these restrictions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#extended-lambda-restrictions).
* functors (like std::plus) or other functions from the STL cannot be used in hipSYCL kernel code. This is because hipSYCL would need to mark these functions as `__device__` to make the CUDA/HIP compiler accept the code, but this is not possible since hipSYCL cannot (and should not) modify STL code [The clang-based source-to-source transformation tool in hipSYCL may use a different STL than the one used by CUDA/HIP compilers, so it is not possible to embed modified STL headers in the transformed source file due to incompatibilities.]
* Due to lack of test hardware, hipSYCL is only compile-tested on AMDs compute stack. If you are using hipSYCL on AMD, feedback is greatly appreciated.
* If the SYCL namespace is fully openend with a `using namespace cl::sycl` statement, bad name collisions can be expected since the SYCL spec requires the existence of SYCL vector types in that namespace with the same name as CUDA/HIP vector types which live in the global namespace.


## News
* 2019/01/18: an implementation of a CPU backend based on OpenMP has been merged into hipSYCL's master branch!
* 2018/12/24: hipSYCL is capable of compiling and running at least some SYCL parallel STL algorithms (more testing ongoing)
* 2018/10/21: The source-to-source transformation step can now also correctly treat header files; the recommended input language for hipSYCL is now regular SYCL.
* 2018/10/07: hipSYCL now comes with an experimental additional source-to-source transformation step during compilation which allows hipSYCL to ingest regular SYCL code.
* 2018/9/24: hipSYCL now compiles cleanly on ROCm as well.

## Building and installing hipSYCL
In order to successfully build and install hipSYCL, a working installation of either CUDA, ROCm (with nvcc/hcc in `$PATH`) or an OpenMP capable compiler is required. At the moment, hipSYCL is tested:
* On NVIDIA: with CUDA 10.0 and gcc 7.3
* On AMD: With the `rocm/rocm-terminal` docker image (only compile testing due to lack of hardware). If you try using hipSYCL with ROCm < 2.0, you will likely run into problems.

### Packages
For Arch Linux users, it is recommended to simply use the `PKGBUILD` provided in `install/archlinux`. A simple `makepkg` in this directory should be enough to build an Arch Linux package.

### Singularity container images
hipSYCL comes with singularity definition files that allow you to create working hipSYCL distributions with a single command, no matter which Linux distribution you are using.

In order to build a ROCm-based hipSYCL singularity container image, simply run:
```
$ sudo singularity build <image-name> install/singularity/hipsycl-rocm.def
```

and for a CUDA based container image, execute:
```
$ sudo singularity build <image-name> install/singularity/hipsycl-cuda.def
```

### Docker container images
Dockerfiles for hipSYCL in conjunction with both CUDA and ROCm are also provided. To build a docker container image for ROCm, run
```
$ sudo docker build install/docker/ROCm
```
and
```
$ sudo docker build install/docker/CUDA
```
for CUDA.

### Manual compilation
hipSYCL depends on:
* python 3 (for the `syclcc` compiler wrapper)
* `cmake`
* `hcc` or `nvcc`
* HIP. On CUDA, it is not necessary to install HIP explicitly since the required headers are bundled with hipSYCL. On AMD, the system-wide HIP installation will be used instead and must be installed and working.
* llvm/clang (with development headers and libraries). LLVM/clang 6 and 7 are supported at the moment.
* the Boost C++ library (only the preprocessor library at the moment)

Once these requirements are met, clone the repository with all submodules:
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

## Compiling software with hipSYCL
hipSYCL provides the `syclcc` compiler driver. `syclcc` will automatically execute the hipSYCL source-to-source transformation toolchain and then invoke either nvcc (for the CUDA backend), hcc (for the ROCm backend) or an OpenMP-capable host compiler (for the CPU backend). If several backends are available,
the backend to be used must be explicitly specified by the user. This can be done either by setting the environment variable
```
export HIPSYCL_PLATFORM=<platform>
```
or with a command line argument for `syclcc`:
```
syclcc --hipsycl-platform=<platform> ...
```
where valid values for `<platform>` are 
* `cuda`, `nvidia` or `nvcc` for CUDA
* `rocm`, `amd`, `hip` or `hcc` for ROCm
* `cpu`, `host` or `hipcpu` for the CPU backend

Note that the CPU backend is at the moment "static", i.e. there's no decision possible at runtime whether to run a kernel on GPU or CPU. Where a kernel is executed depends only on the setting for the hipSYCL platform at compile time.

## Example
The following code adds two vectors:
```cpp
#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

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
                                         [=] (cl::sycl::id<1> tid) {
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



