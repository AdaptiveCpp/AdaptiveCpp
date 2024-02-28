![Project logo](/doc/img/logo/logo-color.png)

# AdaptiveCpp (formerly known as hipSYCL / Open SYCL)

**(Note: This project is currently in progress of changing its name to AdaptiveCpp due to external legal pressure. Documentation and code may still use the older name hipSYCL / Open SYCL)**

AdaptiveCpp is the independent, community-driven modern platform for C++-based heterogeneous programming models targeting CPUs and GPUs from all major vendors. AdaptiveCpp lets applications adapt themselves to all the hardware found in the system. This includes use cases where a single binary needs to be able to target all supported hardware, or utilize hardware from different vendors simultaneously.

It currently supports the following programming models:
1. **SYCL**: At its core is a SYCL implementation that supports many use cases and approaches of implementing SYCL. 
2. **C++ standard parallelism**: Additionally, AdaptiveCpp features experimental support for offloading C++ algorithms from the parallel STL. See [here](doc/stdpar.md) for details on which algorithms can be offloaded. **AdaptiveCpp is currently the only solution that can offload C++ standard parallelism constructs to GPUs from Intel, NVIDIA and AMD -- even from a single binary.**

Supported compilation flows include ([details](doc/compilation.md)):

1. **A generic, single-pass compiler infrastructure that compiles kernels to a unified code representation** that is then lowered at runtime to target devices, providing a high degree of portability, low compilation times, flexibility and extensibility. **AdaptiveCpp is the only major SYCL implementation that supports a single-pass compiler design, where the code is only parsed once for both host and target devices**. Support includes:
   1. NVIDIA CUDA GPUs through PTX;
   2. AMD ROCm GPUs through amdgcn code;
   3. Intel GPUs through SPIR-V (Level Zero);
   4. SPIR-V compatible OpenCL devices supporting Intel USM extensions or fine-grained system SVM (such as Intel's OpenCL implementation for CPUs or GPUs);
   5. The host CPU through LLVM
2. Additionally, **AdaptiveCpp can aggregate existing clang toolchains and augment them with support for SYCL constructs**. This allows for a high degree of interoperability between SYCL and other models such as CUDA or HIP. For example, in this mode, the AdaptiveCpp CUDA and ROCm backends rely on the clang CUDA/HIP frontends that have been augmented by AdaptiveCpp to *additionally* also understand other models like SYCL. This means that the AdaptiveCpp compiler can not only compile SYCL code, but also CUDA/HIP code *even if they are mixed in the same source file*, making all CUDA/HIP features - such as the latest device intrinsics - also available from SYCL code ([details](doc/hip-source-interop.md)). Additionally, vendor-optimized template libraries such as rocPRIM or CUB can also be used with AdaptiveCpp. This allows for highly optimized code paths in SYCL code for specific devices. Support includes:
   1. Any LLVM-supported CPU (including e.g. x86, arm, power etc) through the regular clang host toolchain with dedicated compiler transformation to accelerate SYCL constructs;
   2. NVIDIA CUDA GPUs through the clang CUDA toolchain;
   3. AMD ROCm GPUs through the clang HIP toolchain
3. Or **AdaptiveCpp can be used in library-only compilation flows**. In these compilation flows, AdaptiveCpp acts as a C++ library for third-party compilers. This can have portability advantages or simplify deployment. This includes support:
   1. Any CPU supported by any OpenMP compilers;
   2. NVIDIA GPUs through CUDA and the NVIDIA nvc++ compiler, bringing NVIDIA vendor support and day 1 hardware support to the SYCL ecosystem

The following illustration shows the complete stack and its capabilities to target hardware:
![Compiler stack](/doc/img/stack.png)

Because a program compiled with AdaptiveCpp appears just like any other program written in vendor-supported programming models (like CUDA or HIP) to vendor-provided software, vendor tools such as profilers or debuggers also work well with AdaptiveCpp.

An illustration on how the project fits into the SYCL ecosystem can be found ([here](doc/sycl-ecosystem.md)).


## Performance, extracting performance & benchmarking

See the [AdaptiveCpp performance guide](doc/performance.md).

## Installing and using AdaptiveCpp
* [Building & Installing](doc/installing.md)

In order to compile software with AdaptiveCpp, use `acpp`. `acpp` can be used like a regular compiler, i.e. you can use `acpp -o test test.cpp` to compile your application called `test.cpp` with AdaptiveCpp.

`acpp` accepts both command line arguments and environment variables to configure its behavior (e.g., to select the target to compile for). See `acpp --help` for a comprehensive list of options.

For details and instructions on using AdaptiveCpp in CMake projects, please see the documentation on [using AdaptiveCpp](doc/using-hipsycl.md).

## Documentation
* AdaptiveCpp [design and architecture](doc/architecture.md)
* AdaptiveCpp runtime [specification](doc/runtime-spec.md)
* AdaptiveCpp [compilation model](doc/compilation.md)
* How to use raw HIP/CUDA inside AdaptiveCpp code to create [optimized code paths](doc/hip-source-interop.md)
* A simple SYCL example code for testing purposes can be found [here](doc/examples.md).
* [SYCL Extensions implemented in AdaptiveCpp](doc/extensions.md)
* [Macros used by AdaptiveCpp](doc/macros.md)
* [Environment variables supported by AdaptiveCpp](doc/env_variables.md)


## About the project

Development of AdaptiveCpp is currently primarily led by Heidelberg University, with contributions from a growing community. We see AdaptiveCpp as a community-driven project, in contrast to the many vendor-driven heterogeneous compiler efforts.
AdaptiveCpp not only serves as a research platform, but is also a solution used in production on machines of all scales, including some of the most powerful supercomputers.

### Getting in touch

Join us on [Discord](https://discord.gg/s2p7Vccwh3)!
Alternatively, open a discussion or issue in this repository.

### Contributing to AdaptiveCpp

We encourage contributions and are looking forward to your pull request! Please have a look at [CONTRIBUTING.md](CONTRIBUTING.md). If you need any guidance, please just open an issue and we will get back to you shortly.

If your institution or organization is considering to support the AdaptiveCpp development in some official capacity, we are always happy to discuss collaborations and to broaden the developer community. Please do reach out :-)

If you are a student at Heidelberg University and wish to work on AdaptiveCpp, please get in touch with us. There are various options possible and we are happy to include you in the project :-)

### Citing AdaptiveCpp

AdaptiveCpp is a production platform for heterogeneous computing, but also a research project. As such, if you use AdaptiveCpp in your research, we kindly request that you cite one of the following publications, depending on your focus:

* A general overview, SYCL 2020, performance and the relationship with oneAPI: *Aksel Alpay, Bálint Soproni, Holger Wünsche, and Vincent Heuveline. 2022. Exploring the possibility of a hipSYCL-based implementation of oneAPI. In International Workshop on OpenCL (IWOCL'22). Association for Computing Machinery, New York, NY, USA, Article 10, 1–12. https://doi.org/10.1145/3529538.3530005*
* The generic single-pass compiler: *Aksel Alpay and Vincent Heuveline. 2023. One Pass to Bind Them: The First Single-Pass SYCL Compiler with Unified Code Representation Across Backends. In Proceedings of the 2023 International Workshop on OpenCL (IWOCL '23). Association for Computing Machinery, New York, NY, USA, Article 7, 1–12. https://doi.org/10.1145/3585341.3585351*
* Our CPU compiler: *Joachim Meyer, Aksel Alpay, Sebastian Hack, Holger Fröning, and Vincent Heuveline. 2023. Implementation Techniques for SPMD Kernels on CPUs. In Proceedings of the 2023 International Workshop on OpenCL (IWOCL '23). Association for Computing Machinery, New York, NY, USA, Article 1, 1–12. https://doi.org/10.1145/3585341.3585342*
* The original talk and the idea of implementing SYCL on non-OpenCL backends: *Aksel Alpay and Vincent Heuveline. 2020. SYCL beyond OpenCL: The architecture, current state and future direction of hipSYCL. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 8, 1. DOI:https://doi.org/10.1145/3388333.3388658*

(The latter is a talk and available [online](https://www.youtube.com/watch?v=kYrY80J4ZAs). Note that some of the content in this talk is outdated by now)

### Acknowledgements

We gratefully acknowledge [contributions](https://github.com/illuhad/hipSYCL/graphs/contributors) from the community.

