![Project logo](/doc/img/logo/logo-color.png)

# hipSYCL - a SYCL implementation for CPUs and GPUs

hipSYCL is a modern SYCL implementation targeting CPUs and GPUs, with a focus on leveraging existing toolchains such as CUDA or HIP. hipSYCL currently targets the following devices:
* Any CPU via OpenMP
* NVIDIA GPUs via CUDA
* AMD GPUs via HIP/ROCm
* Intel GPUs via oneAPI Level Zero and SPIR-V (*highly* experimental and WIP!)

hipSYCL supports compiling source files into a single binary that can run on all these backends when building against appropriate clang distributions. More information about the [compilation flow can be found here](doc/compilation.md).

The runtime architecture of hipSYCL consists of the main library `hipSYCL-rt`, as well as independent, modular plugin libraries for the individual backends:
![Runtime architecture](/doc/img/runtime.png)

hipSYCL's compilation and runtime design allows hipSYCL to **effectively aggregate multiple toolchains that are otherwise incompatible, making them accessible with a single SYCL interface.**

The philosophy behind hipSYCL is to leverage such existing toolchains as much as possible. This brings not only maintenance and stability advantages, but enables performance on par with those established toolchains by design, and also allows for maximum interoperability with existing compute platforms.
For example, the hipSYCL CUDA and ROCm backends rely on the clang CUDA/HIP frontends that have been augmented by hipSYCL to *additionally* also understand SYCL code. This means that the hipSYCL compiler can not only compile SYCL code, but also CUDA/HIP code *even if they are mixed in the same source file*, making all CUDA/HIP features - such as the latest device intrinsics - also available from SYCL code ([details](doc/hip-source-interop.md)). Additionally, vendor-optimized template libraries such as rocPRIM or CUB can also be used with hipSYCL. Consequently, hipSYCL allows for **highly optimized code paths in SYCL code for specific devices**.

Because a SYCL program compiled with hipSYCL looks just like any other CUDA or HIP program to vendor-provided software, vendor tools such as profilers or debuggers also work well with hipSYCL.

The following image illustrates how hipSYCL fits into the wider SYCL implementation ecosystem:
<img src="doc/img/sycl-targets.png" width=80% height=80%>

## About the project

While hipSYCL started its life as a hobby project, development is now led and funded by Heidelberg University. hipSYCL not only serves as a research platform, but is also a solution used in production on machines of all scales, including some of the most powerful supercomputers.

### Contributing to hipSYCL

We encourage contributions and are looking forward to your pull request! Please have a look at [CONTRIBUTING.md](CONTRIBUTING.md). If you need any guidance, please just open an issue and we will get back to you shortly.

If you are a student at Heidelberg University and wish to work on hipSYCL, please get in touch with us. There are various options possible and we are happy to include you in the project :-)

### Citing hipSYCL

hipSYCL is a research project. As such, if you use hipSYCL in your research, we kindly request that you cite:

*Aksel Alpay and Vincent Heuveline. 2020. SYCL beyond OpenCL: The architecture, current state and future direction of hipSYCL. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 8, 1. DOI:https://doi.org/10.1145/3388333.3388658*

(This is a talk and available [online](https://www.youtube.com/watch?v=kYrY80J4ZAs). Note that some of the content in this talk is outdated by now)

### Acknowledgements

We gratefully acknowledge [contributions](https://github.com/illuhad/hipSYCL/graphs/contributors) from the community.

## Performance

hipSYCL has been repeatedly shown to deliver very competitive performance compared to other SYCL implementations or proprietary solutions like CUDA. See for example:

* *Sohan Lal, Aksel Alpay, Philip Salzmann, Biagio Cosenza, Nicolai Stawinoga, Peter Thoman, Thomas Fahringer, and Vincent Heuveline. 2020. SYCL-Bench: A Versatile Single-Source Benchmark Suite for Heterogeneous Computing. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 10, 1. DOI:https://doi.org/10.1145/3388333.3388669*
* *Brian Homerding and John Tramm. 2020. Evaluating the Performance of the hipSYCL Toolchain for HPC Kernels on NVIDIA V100 GPUs. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 16, 1–7. DOI:https://doi.org/10.1145/3388333.3388660*
* *Tom Deakin and Simon McIntosh-Smith. 2020. Evaluating the performance of HPC-style SYCL applications. In Proceedings of the International Workshop on OpenCL (IWOCL ’20). Association for Computing Machinery, New York, NY, USA, Article 12, 1–11. DOI:https://doi.org/10.1145/3388333.3388643*


### Benchmarking hipSYCL

When targeting the CUDA or HIP backends, hipSYCL just massages the AST slightly to get `clang -x cuda` and `clang -x hip` to accept SYCL code. hipSYCL is not involved in the actual code generation. Therefore *any significant deviation in kernel performance compared to clang-compiled CUDA or clang-compiled HIP is unexpected.*

As a consequence, if you compare it to other llvm-based compilers please make sure to compile hipSYCL against the same llvm version. Otherwise you would effectively be simply comparing the performance of two different LLVM versions. This is in particular true when comparing it to clang CUDA or clang HIP.


## Current state
hipSYCL is not yet a fully conformant SYCL implementation, although many SYCL programs already work with hipSYCL.
* SYCL 2020 [feature support matrix](https://github.com/hipSYCL/featuresupport)
* A (likely incomplete) list of [limitations](doc/limitations.md) for older SYCL 1.2.1 features
* A (also incomplete) timeline showing development [history](doc/history.md)

## Hardware and operating system support

Supported hardware:
* Any CPU for which a C++17 OpenMP compiler exists
* NVIDIA CUDA GPUs. Note that clang, which hipSYCL relies on, may not always support the very latest CUDA version which may sometimes impact support for *very* new hardware. See the [clang documentation](https://www.llvm.org/docs/CompileCudaWithLLVM.html) for more details.
* AMD GPUs that are [supported by ROCm](https://github.com/RadeonOpenCompute/ROCm#hardware-support)

Operating system support currently strongly focuses on Linux. On Mac, only the CPU backend is expected to work. Windows support with CPU and CUDA backends is experimental, see [Using hipSYCL on Windows](https://github.com/illuhad/hipSYCL/wiki/Using-hipSYCL-on-Windows).

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

The full documentation of syclcc and hints for the CMake integration can be found in [using hipSYCL](doc/using-hipsycl.md).

## Documentation
* hipSYCL [design and architecture](doc/architecture.md)
* hipSYCL runtime [specification](doc/runtime-spec.md)
* hipSYCL [compilation model](doc/compilation.md)
* How to use raw HIP/CUDA inside hipSYCL code to create [optimized code paths](doc/hip-source-interop.md)
* A simple SYCL example code for testing purposes can be found [here](doc/examples.md).
* [SYCL Extensions implemented in hipSYCL](doc/extensions.md)
* [Macros used by hipSYCL](doc/macros.md)
* [Environment variables supported by hipSYCL](doc/env_variables.md)



