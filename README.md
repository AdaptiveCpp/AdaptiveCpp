**Note: The master branch is deprecated, all code has been removed from it. Users are encouraged to switch to `develop` or `stable` branches. Only documentation is archived and retained on the master branch to avoid breaking links from external sites that references these documents.**

![Project logo](/doc/img/logo/logo-color.png)

# hipSYCL - a SYCL implementation for CPUs and GPUs
[![Build Status](https://travis-ci.com/illuhad/hipSYCL.svg?branch=master)](https://travis-ci.com/illuhad/hipSYCL)

hipSYCL is a modern SYCL implementation targeting CPUs and GPUs, with a focus on leveraging existing toolchains such as CUDA or HIP. hipSYCL currently targets the following devices:
* Any CPU via OpenMP
* NVIDIA GPUs via CUDA
* AMD GPUs via HIP/ROCm

The following image illustrates how hipSYCL fits into the wider SYCL implementation ecosystem:
![SYCL implementations](/doc/img/sycl-targets.png)

The philosophy behind hipSYCL is to leverage existing toolchains as much as possible. This brings not only maintenance and stability advantages, but enables performance on par with those established toolchains by design, and allows for maximum interoperability with existing compute platforms.
For example, the hipSYCL CUDA and ROCm backends rely on the clang CUDA/HIP frontends that have been augmented by hipSYCL to *additionally* also understand SYCL code. This means that the hipSYCL compiler can not only compile SYCL code, but also CUDA/HIP code *even if they are mixed in the same source file*, making all CUDA/HIP features - such as the latest device intrinsics - also available from SYCL code ([details](doc/hip-source-interop.md)). Additionally, vendor-optimized template libraries such as rocPRIM or CUB can also be used with hipSYCL. Consequently, hipSYCL allows for *highly optimized code paths in SYCL code for specific devices*.

Because a SYCL program compiled with hipSYCL looks just like any other CUDA or HIP program to vendor-provided software, vendor tools such as profilers or debuggers also work well with hipSYCL.

## About the project

While hipSYCL started its life as a hobby project, development is now led and funded by Heidelberg University. hipSYCL not only serves as a research platform, but is also a solution used in production on machines of all scales, including some of the most powerful supercomputers.

### Contributing to hipSYCL

We encourage contributions and are looking forward to your pull request! Please have a look at [CONTRIBUTING.md](CONTRIBUTING.md). If you need any guidance, please just open an issue and we will get back to you shortly.

If you are a student at Heidelberg University and wish to work on hipSYCL, please get in touch with us. There are various options possible and we are happy to include you in the project :-)

### Citing hipSYCL

hipSYCL is a research project. As such, if you use hipSYCL in your research, we kindly request that you cite:

*Aksel Alpay and Vincent Heuveline. 2020. SYCL beyond OpenCL: The architecture, current state and future direction of hipSYCL. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 8, 1. DOI:https://doi.org/10.1145/3388333.3388658*

### Acknowledgements

We gratefully acknowledge [contributions](https://github.com/illuhad/hipSYCL/graphs/contributors) from the community.

## Performance

hipSYCL has been repeatedly shown to deliver very competitive performance compared to other SYCL implementations or proprietary solutions like CUDA. See for example:

* *Sohan Lal, Aksel Alpay, Philip Salzmann, Biagio Cosenza, Nicolai Stawinoga, Peter Thoman, Thomas Fahringer, and Vincent Heuveline. 2020. SYCL-Bench: A Versatile Single-Source Benchmark Suite for Heterogeneous Computing. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 10, 1. DOI:https://doi.org/10.1145/3388333.3388669*
* *Brian Homerding and John Tramm. 2020. Evaluating the Performance of the hipSYCL Toolchain for HPC Kernels on NVIDIA V100 GPUs. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 16, 1–7. DOI:https://doi.org/10.1145/3388333.3388660*
* *Tom Deakin and Simon McIntosh-Smith. 2020. Evaluating the performance of HPC-style SYCL applications. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 12, 1–11. DOI:https://doi.org/10.1145/3388333.3388643*


## Current state
hipSYCL is not yet a fully conformant SYCL implementation, although many SYCL programs already work with hipSYCL.
* A (likely incomplete) list of current [limitations](doc/limitations.md)
* A (also incomplete) timeline showing development [history](doc/history.md)

## Hardware and operating system support

Supported hardware:
* Any CPU for which a C++17 OpenMP compiler exists
* NVIDIA CUDA GPUs. Note that clang, which hipSYCL relies on, may not always support the very latest CUDA version which may sometimes impact support for *very* new hardware. See the [clang documentation](https://www.llvm.org/docs/CompileCudaWithLLVM.html) for more details.
* AMD GPUs that are [supported by ROCm](https://github.com/RadeonOpenCompute/ROCm#hardware-support)

Operating system support currently strongly focuses on Linux. On Mac, only the CPU backend is expected to work. Windows is currently not supported.

## Installing and using hipSYCL
* [Building & Installing](doc/installing.md)

In order to compile software with hipSYCL, use `syclcc` which automatically adds all required compiler arguments to the CUDA/HIP compiler. `syclcc` can be used like a regular compiler, i.e. you can use `syclcc -o test test.cpp` to compile your SYCL application called `test.cpp` with hipSYCL.

`syclcc` accepts both command line arguments and environment variables to configure its behavior (e.g., to select the target platform CUDA/ROCm/CPU to compile for). See `syclcc --help` for a comprehensive list of options.

When targeting a GPU, you will need to provide a target GPU architecture. The expected formats are defined by clang CUDA/HIP. Examples:
* `sm_52`: NVIDIA Maxwell GPUs
* `sm_60`: NVIDIA Pascal GPUs
* `sm_70`: NVIDIA Volta GPUs
* `gfx900`: AMD Vega 10 GPUs
* `gfx906`: AMD Vega 20 GPUs

## Documentation
* hipSYCL [compilation model](doc/compilation.md)
* How to use raw HIP/CUDA inside hipSYCL code to create [optimized code paths](doc/hip-source-interop.md)
* A simple SYCL example code for testing purposes can be found [here](doc/examples.md).
* [SYCL Extensions implemented in hipSYCL](doc/extensions.md)
* [Macros used by hipSYCL](doc/macros.md)



