
# Building and installing AdaptiveCpp

## Manual installation (Linux)

### Software dependencies
In order to successfully build and install AdaptiveCpp, the following dependencies must be installed for all backends:

* python 3 (for the `acpp` compiler driver)
* `cmake`
* the Boost C++ libraries (in particular `boost.fiber`, `boost.context` and for the unit tests `boost.test`)
  * it may be helpful to set the `BOOST_ROOT` `cmake` variable to the path to the root directory of Boost you wish to use if `cmake` does not find it automatically
  * **Note for boost 1.78 users:** There seems to be a bug in the build system for boost 1.78, causing the compiled fiber and context libraries not to be copied to the installation directory. You will have to copy these libraries manually to the installation directory. In binary packages from some distribution repositories this issue is fixed. You might be only affected when building boost manually from source.

In addition, the various supported [compilation flows](compilation.md) and programming models have additional requirements:

#### Compilation flows

| Compilation flow | Target hardware | Short description | Requirements |
|------------------|-------------------|-------------------|-------------------|
| `omp.library-only` | Any CPU | OpenMP CPU backend | Any OpenMP compiler |
| `omp.accelerated` | Any CPU supported by LLVM | OpenMP CPU backend (compiler-accelerated)| LLVM >= 11 |
| `cuda.integrated-multipass` | NVIDIA GPUs | CUDA backend (clang)| CUDA >= 10, LLVM >= 10 |
| `cuda.explicit-multipass` | NVIDIA GPUs | CUDA backend (clang, can be targeted simultaneously with other backends) | CUDA >= 10, LLVM 11 or 13+ |
| `cuda-nvcxx` | NVIDIA GPUs | CUDA backend (nvc++) | Latest NVIDIA HPC SDK |
| `hip.integrated-multipass` | AMD GPUs (supported by ROCm) | HIP backend (clang) | ROCm >= 4.0, LLVM >= 10 |
| `spirv` | Intel GPUs | SPIR-V/Level Zero backend | Level Zero driver and loader, clang with SYCL patches (e.g DPC++) |
| `generic` | NVIDIA, AMD, Intel GPUs, OpenCL SPIR-V devices | Generic single-pass compiler | LLVM >= 14. When dispatching kernels to AMD hardware, ROCm >= 5.3 is recommended. When dispatching to NVIDIA, clang needs nvptx64 backend enabled. AdaptiveCpp runtime backends for the respective target hardware need to be available. |

#### Models

* SYCL: (No SYCL-specific requirements)
* C++ standard parallelism: See [here](stdpar.md) for dependencies.

Please make sure to read the instructions below for the dependencies that apply to your use case.

#### LLVM (skip if you only want flows without LLVM dependency)

Follow [these](install-llvm.md) instructions.

#### CUDA (skip if you don't need CUDA support)

Follow [these](install-cuda.md) instructions

#### ROCm (skip if you don't need ROCm support)

Follow [these](install-rocm.md) instructions

#### SPIR-V/Level Zero (skip if you don't need SPIR-V/Level Zero support)

Follow [these](install-spirv.md) instructions.

#### SPIR-V/OpenCL (skip if you don't need SPIR-V/OpenCL support)

Follow [these](install-ocl.md) instructions.

#### Building and installing 

Once the software requirements mentioned above are met, clone the repository:
```
$ git clone https://github.com/OpenSYCL/OpenSYCL
```
Then, create a build directory and compile AdaptiveCpp. As described below, some backends and compilation flows must be configured with specific cmake arguments which should be passed during the cmake step.

```
$ cd <build directory>
$ cmake -DCMAKE_INSTALL_PREFIX=<installation prefix> <more optional options, e.g. to configure the LLVM dependency> <AdaptiveCpp source directory>
$ make install
```

The default installation prefix is `/usr/local`. Change this to your liking.
**Note: AdaptiveCpp needs to be installed to function correctly; don't replace "make install" with just "make"!**

##### CMake options to configure the AdaptiveCpp build

###### General
*  `-DCMAKE_CXX_COMPILER` should be pointed to the C++ compiler to compile AdaptiveCpp with. Note that this also sets the default C++ compiler for the CPU backend when using acpp once AdaptiveCpp is installed. This can however also be modified later using `HIPSYCL_CPU_CXX`.

###### omp.library-only

* `-DCMAKE_CXX_COMPILER` can be used to set the default OpenMP compiler.

###### omp.accelerated

* `-DWITH_ACCELERATED_CPU=OFF/ON` can be used to explicitly disable/enable CPU acceleration. Support for CPU acceleration is enabled by default when enabling the LLVM dependency, and LLVM is sufficiently new.

###### cuda.*

* See the CUDA [installation instructions](install-cuda.md) instructions (section on clang).

###### cuda-nvcxx

* See the CUDA [installation instructions](install-cuda.md) instructions (section on nvc++).

###### hip.*

* See the ROCm [installation instructions](install-rocm.md) instructions.

###### spirv

* No specific cmake flags are currently available.

## Manual installation (Mac)

On Mac, only the CPU backends are supported. The required steps are analogous to Linux.

## Manual installation (Windows)

For experimental building on Windows (CPU and CUDA backends) see the corresponding [wiki](https://github.com/OpenSYCL/OpenSYCL/wiki/Using-hipSYCL-on-Windows).
The `omp.accelerated` CPU compilation flow is unsupported on Windows.

## Repositories (Linux)

**Note: The software repositories mentioned below are outdated and in the process of being restructured. They do not contain modern AdaptiveCpp versions.**

Another way to install AdaptiveCpp is to use our repositories. We provide repositories for several distributions (currently Ubuntu 18.04, CentOS 7, Arch Linux). A description of the repositories is available [here](../install/scripts/README.md#installing-from-repositories)

Our repositories cover the *entire software stack*, i.e. they include a compatible clang/LLVM distribution and ROCm stacks. The following packages are available:
* `hipSYCL` - contains the actual AdaptiveCpp libraries, tools and headers
* `hipSYCL-base` - contains the LLVM/clang stack used by AdaptiveCpp. Installation of this package is mandatory.
* `hipSYCL-rocm` - contains a ROCm stack. This package is only required if you wish to target AMD ROCm GPUs.
* `hipSYCL-nightly` - built from the current develop branch every day.
* `hipSYCL-base-nightly` - contains the LLVM/clang stack for the nightly AdaptiveCpp packages
* `hipSYCL-rocm-nightly` - contains a ROCm stack compatible with the nightly AdaptiveCpp packages

**Note: For legal reasons, we do not redistribute the hipSYCL-cuda package** This package is only required if you wish to target CUDA GPUs. You will either have to create a CUDA package using `install/scripts/packaging/make-<distribution>-cuda-pkg.sh` or you can install CUDA directly using the `install/scripts/install-cuda.sh` script.


## Installation scripts

**Note: The scripts are outdated and in process of being restructured. They may or may not work with recent AdaptiveCpp versions**
We also provide scripts for packaging AdaptiveCpp and its dependencies. For more information on packaging and how to create your own AdaptiveCpp packages, please see the [documentation](../install/scripts/README.md).
