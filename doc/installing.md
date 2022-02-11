
# Building and installing hipSYCL

## Manual installation (Linux)

### Software dependencies
In order to successfully build and install hipSYCL, the following dependencies must be installed for all backends:

* python 3 (for the `syclcc` and `syclcc-clang` compiler wrappers)
* `cmake`
* the Boost C++ libraries (in particular `boost.fiber`, `boost.context` and for the unit tests `boost.test`)
  * it may be helpful to set the `BOOST_ROOT` `cmake` variable to the path to the root directory of Boost you wish to use if `cmake` does not find it automatically

In addition, the various supported compilation flows have additional requirements (see [here](compilation.md) for more information on available compilation flows):

| Compilation flow | Target hardware | Short description | Requirements |
|------------------|-------------------|-------------------|-------------------|
| `omp` | Any CPU | OpenMP CPU backend | Any OpenMP compiler |
| `omp.accelerated` | Any CPU supported by LLVM | OpenMP CPU backend (compiler-accelerated)| LLVM >= 11 |
| `cuda.integrated-multipass` | NVIDIA GPUs | CUDA backend (clang)| CUDA >= 10, LLVM >= 10 |
| `cuda.explicit-multipass` | NVIDIA GPUs | CUDA backend (clang, can be targeted simultaneously with other backends) | CUDA >= 10, LLVM 11 or 13+ |
| `cuda-nvcxx` | NVIDIA GPUs | CUDA backend (nvc++) | Latest NVIDIA HPC SDK |
| `hip.integrated-multipass` | AMD GPUs (supported by ROCm) | HIP backend (clang) | ROCm >= 4.0, LLVM >= 10 |
| `spirv` | Intel GPUs | SPIR-V/Level Zero backend | Level Zero driver and loader, clang with SYCL patches (e.g DPC++) |

Please make sure to read the instructions below for the dependencies that apply to your use case.

#### LLVM (skip if you only want flows without LLVM dependency)

Follow [these](install-llvm.md) instructions.

#### CUDA (skip if you don't need CUDA support)

Follow [these](install-cuda.md) instructions

#### ROCm (skip if you don't need ROCm support)

Follow [these](install-rocm.md) instructions

#### SPIR-V/Level Zero (skip if you don't need SPIR-V/Level Zero support)

Follow [these](install-spirv.md) instructions.

#### Building and installing 

Once the software requirements mentioned above are met, clone the repository:
```
$ git clone https://github.com/illuhad/hipSYCL
```
Then, create a build directory and compile hipSYCL:
```
$ cd <build directory>
$ cmake -DCMAKE_INSTALL_PREFIX=<installation prefix> <more optional options, e.g. to configure the LLVM dependency> <hipSYCL source directory>
$ make install
```
The default installation prefix is `/usr/local`. Change this to your liking.
**Note: hipSYCL needs to be installed to function correctly; don't replace "make install" with just "make"!**

A `cmake` variable that maybe useful to set is `CMAKE_CXX_COMPILER` which should be pointed to the C++ compiler to compile hipSYCL with. Note that this also sets the default C++ compiler for the CPU backend when using syclcc once hipSYCL is installed. This can however also be modified later using `HIPSYCL_CPU_CXX`.

## Manual installation (Mac)

On Mac, only the CPU backends are supported. The required steps are analogous to Linux.

## Manual installation (Windows)

For experimental building on Windows (CPU and CUDA backends) see the corresponding [wiki](https://github.com/illuhad/hipSYCL/wiki/Using-hipSYCL-on-Windows).

## Repositories (Linux)

Another way to install hipSYCL is to use our repositories. We provide repositories for several distributions (currently Ubuntu 18.04, CentOS 7, Arch Linux). A description of the repositories is available [here](../install/scripts/README.md#installing-from-repositories)

Our repositories cover the *entire software stack*, i.e. they include a compatible clang/LLVM distribution and ROCm stacks. The following packages are available:
* `hipSYCL` - contains the actual hipSYCL libraries, tools and headers
* `hipSYCL-base` - contains the LLVM/clang stack used by hipSYCL. Installation of this package is mandatory.
* `hipSYCL-rocm` - contains a ROCm stack. This package is only required if you wish to target AMD ROCm GPUs.
* `hipSYCL-nightly` - built from the current develop branch every day.
* `hipSYCL-base-nightly` - contains the LLVM/clang stack for the nightly hipSYCL packages
* `hipSYCL-rocm-nightly` - contains a ROCm stach compatible with the nighlty hipSYCL packages

**Note: For legal reasons, we do not redistribute the hipSYCL-cuda package** This package is only required if you wish to target CUDA GPUs. You will either have to create a CUDA package using `install/scripts/packaging/make-<distribution>-cuda-pkg.sh` or you can install CUDA directly using the `install/scripts/install-cuda.sh` script.


## Installation scripts
We also provide scripts for packaging hipSYCL and its dependencies. For more information on packaging and how to create your own hipSYCL packages, please see the [documentation](../install/scripts/README.md).
