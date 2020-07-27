
# Building and installing hipSYCL

## Repositories
The easiest way to install hipSYCL is to use our repositories. We provide repositories for several distributions (currently Ubuntu 18.04, CentOS 7, Arch Linux). A description of the repositories is available [here](../install/scripts/README.md#installing-from-repositories)

Our repositories cover the *entire software stack*, i.e. they include a compatible clang/LLVM distribution and ROCm stacks. The following packages are available:
* `hipSYCL` - contains the actual hipSYCL libraries, tools and headers
* `hipSYCL-base` - contains the LLVM/clang stack used by hipSYCL. Installation of this package is mandatory.
* `hipSYCL-rocm` - contains a ROCm stack. This package is only required if you wish to target AMD ROCm GPUs.

**Note: For legal reasons, we do not redistribute the hipSYCL-cuda package** This package is only required if you wish to target CUDA GPUs. You will either have to create a CUDA package using `install/scripts/packaging/make-<distribution>-cuda-pkg.sh` or you can install CUDA directly using the `install/scripts/install-cuda.sh` script.


## Installation scripts
We also provide scripts for packaging hipSYCL and its dependencies. For more information on packaging and how to create your own hipSYCL packages, please see the [documentation](../install/scripts/README.md).

## Manual installation

### Software dependencies
In order to successfully build and install hipSYCL, the following major requirements must be met:

* **LLVM and clang >= 8** must be installed, including development files.
* *For the CUDA backend*: 
  * **CUDA >= 9.2** must be installed.
  * hipSYCL requires clang for CUDA compilation. clang usually produces CUDA programs with very competitive performance compared to nvcc. For more information on compiling CUDA with clang, see [here](http://llvm.org/docs/CompileCudaWithLLVM.html).
  * Read more about [compatible clang and CUDA versions for the CUDA backend](install-cuda.md).
* *For the ROCm backend*: 
  * **ROCm >= 2.0** must be installed. 
  * hipSYCL requires clang for ROCm compilation. While AMD's clang forks can in principle be used, regular clang is usually easier to set up. Read more about [compatible clang versions for the ROCm backend](install-rocm.md).
* *For the CPU backend*: Any C++ compiler with **OpenMP** support should do.
* python 3 (for the `syclcc` and `syclcc-clang` compiler wrappers)
* `cmake`
* hipSYCL also depends on HIP, but already comes bundled with a version of the HIP headers for the CUDA backend. The ROCm backend will use the HIP installation that is installed as part of ROCm.
* the Boost C++ libraries (in particular `boost.fiber`, `boost.context` and for the unit tests `boost.test`)

If hipSYCL does not automatically configure the build for the desired clang/LLVM installation, the following cmake variables can be used to point hipSYCL to the right one:
* `LLVM_DIR` must be pointed to your LLVM installation, specifically, the **subdirectory containing the LLVM cmake files**
* `CLANG_EXECUTABLE_PATH` must be pointed to the `clang++` executable from this LLVM installation.
* `CLANG_INCLUDE_PATH` must be pointed to the clang internal header directory. Typically, this is something like `$LLVM_INSTALL_PREFIX/include/clang/<llvm-version>/include`.

#### Building and installing 

Once the software requirements mentioned above are met, clone the repository with all submodules:
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