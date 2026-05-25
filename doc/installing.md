# Building and Installing AdaptiveCpp

This guide provides instructions for building and installing AdaptiveCpp from source. 

## Quick Start (Linux)

For most users on a modern Linux distribution (Ubuntu, Debian, Fedora, etc.), a standard installation with LLVM support is recommended.

1.  **Install base dependencies**: `cmake`, `python3`, `libboost-test-dev`.
2.  **Install LLVM**: We recommend **LLVM 15 to 20**.
    ```bash
    # Ubuntu example for LLVM 20
    sudo apt install clang-20-dev llvm-20-dev lld-20 libomp-20-dev
    ```
3.  **Clone and Build**:
    ```bash
    git clone https://github.com/AdaptiveCpp/AdaptiveCpp
    cd AdaptiveCpp && mkdir build && cd build
    cmake -DCMAKE_INSTALL_PREFIX=/opt/acpp ..
    make -j$(nproc) install
    ```

---

## Operating System Support

| OS | Level of Support | Supported Backends |
| :--- | :--- | :--- |
| **Linux** | **Primary** | All (CPU, CUDA, ROCm, OpenCL, Level Zero) |
| **macOS** | Supported | CPU, [Metal (Experimental)](install-metal.md) |
| **Windows** | Experimental | CPU, CUDA (via [Advanced Build](advanced-builds.md)) |

> [!TIP]
> For **Windows** users, we provide [nightly binaries](https://nightly.link/AdaptiveCpp/AdaptiveCpp/workflows/windows-acppllvm/develop/AdaptiveCpp-LLVM20-Win.zip) for the `develop` branch.

---

## Prerequisites (Linux)

### Core Dependencies
- **Python 3**: Required for the `acpp` compiler driver.
- **CMake**: Version 3.13 or newer.
- **Boost Libraries**: Only `boost.test` is strictly required for unit tests.
    - Set `BOOST_ROOT` if CMake fails to find it.
    - **Note for Boost 1.87/1.88**: These versions have known issues with `Boost.Math` in SYCL paths. Use versions < 1.87 or >= 1.89.

### Backend-Specific Requirements

```
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/your/desired/install/location ..
make install
```
If it does not find some backends or does not pick up the right LLVM, please look at the documentation for the individual components linked below.


### Advanced installation

Advanced users may want to customize their installation more, or use features that are not so commonly used. The following sections describe requirements for individual components in more detail.

#### Compilation flows

| Compilation flow | Target hardware | Short description | Requirements |
|------------------|-------------------|-------------------|-------------------|
| `omp.library-only` | Any CPU | OpenMP CPU backend | Any OpenMP compiler |
| `omp.accelerated` | Any CPU supported by LLVM | OpenMP CPU backend (compiler-accelerated)| LLVM* >= 15 and LLVM* <= 21|
| `cuda.integrated-multipass` | NVIDIA GPUs | CUDA backend (clang)| CUDA >= 10, LLVM* >= 15 and LLVM* <= 21|
| `cuda.explicit-multipass` | NVIDIA GPUs | CUDA backend (clang, can be targeted simultaneously with other backends) | CUDA >= 10, LLVM* >= 15 and LLVM* <= 21 |
| `cuda-nvcxx` | NVIDIA GPUs | CUDA backend (nvc++) | Latest NVIDIA HPC SDK |
| `hip.integrated-multipass` | AMD GPUs (supported by ROCm) | HIP backend (clang) | ROCm >= 4.0, LLVM* >= 15 and LLVM* <= 21 |
| `generic` | NVIDIA, AMD, Intel GPUs, OpenCL SPIR-V devices, Apple GPUs (experimental) | Generic single-pass compiler | LLVM* >= 15 and LLVM* <= 21. When dispatching kernels to AMD hardware, ROCm >= 5.3 is recommended and LLVM must be <= the ROCm LLVM version. When dispatching to NVIDIA, clang needs nvptx64 backend enabled. AdaptiveCpp runtime backends for the respective target hardware need to be available. For Apple GPUs, see [Metal installation instructions](install-metal.md). |

\* AdaptiveCpp does not support development versions of LLVM, only official releases are supported.

Note: Building against `libc++` instead of `libstdc++` is only expected to work for the `generic` target. Additionally, AdaptiveCpp must have been built using the same standard library that the user code is linked against.
`libc++` is currently not supported for the C++ standard parallelism offloading model.

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

#### Metal (skip if you don't need Metal support, macOS only)

Follow [these](install-metal.md) instructions. Note that this is an experimental backend.

#### Building and installing 

Once the software requirements mentioned above are met, clone the repository:
```
$ git clone https://github.com/AdaptiveCpp/AdaptiveCpp
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
* `-DACPP_COMPILER_FEATURE_PROFILE` can be used to configure the desired degree of compiler support. Supported values:
    * `full` (default and recommended): Enables all AdaptiveCpp features, requires a compatible LLVM installation as described [here](install-llvm.md). This is recommended for both functionality and performance.
    * `minimal`: Only enables the older interoperability-focused compilation flows for CUDA and HIP (`--acpp-targets=cuda` and `--acpp-targets=hip`). No OpenCL or Level Zero support, no C++ standard parallelism offloading support, no generic JIT compiler (`generic` target), no compiler acceleration for SYCL constructs on CPU device. **Should only be selected in specific circumstances.**
    * `none`: Disables all compiler support and dependencies on LLVM. In addition to `minimal`, also disables the support for `--acpp-targets=cuda` and `--acpp-targets=hip`. In this mode, AdaptiveCpp operates purely as a library for third-party compilers. **Should only be selected in specific circumstances.**

###### generic

* Requires `-DACPP_COMPILER_FEATURE_PROFILE=full`

###### omp.library-only

* `-DCMAKE_CXX_COMPILER` can be used to set the default OpenMP compiler.

###### omp.accelerated

* Requires `-DACPP_COMPILER_FEATURE_PROFILE=full`

---

## Building from Source

Once requirements are met, use the following standard CMake workflow:

```bash
git clone https://github.com/AdaptiveCpp/AdaptiveCpp
cd AdaptiveCpp && mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/your/install/path [OPTIONS] ..
make install
```

> [!IMPORTANT]
> **Always run `make install`**. AdaptiveCpp requires a proper installation tree to function; it cannot be run directly from the build directory.

### Common CMake Options

- `-DCMAKE_CXX_COMPILER`: The compiler used to build AdaptiveCpp and the default for CPU backends.
- `-DACPP_COMPILER_FEATURE_PROFILE`:
    - `full` (Default): Enables all features (requires LLVM).
    - `minimal`: Enables only older CUDA/HIP flows (reduced dependencies).
    - `none`: Pure library mode (no compiler support).

#### Manually Enabling/Disabling Backends
AdaptiveCpp attempts to auto-detect supported backends. You can override this behavior using the following flags:

| Backend | Enable Flag | Disable Flag |
| :--- | :--- | :--- |
| **CUDA** | `-DWITH_CUDA_BACKEND=ON` | `-DWITH_CUDA_BACKEND=OFF` |
| **ROCm** | `-DWITH_ROCM_BACKEND=ON` | `-DWITH_ROCM_BACKEND=OFF` |
| **OpenCL** | `-DWITH_OPENCL_BACKEND=ON` | `-DWITH_OPENCL_BACKEND=OFF` |
| **Level Zero** | `-DWITH_LEVEL_ZERO_BACKEND=ON` | `-DWITH_LEVEL_ZERO_BACKEND=OFF` |
| **Metal** | `-DWITH_METAL_BACKEND=ON` | `-DWITH_METAL_BACKEND=OFF` |

---

## Detailed Backend Instructions

For specific hardware and driver configurations, see the following dedicated guides:

- [**LLVM & Clang**](install-llvm.md): Core compiler dependency for most flows.
- [**NVIDIA CUDA**](install-cuda.md): For NVIDIA GPU support.
- [**AMD ROCm**](install-rocm.md): For AMD GPU support.
- [**Intel / SPIR-V**](install-spirv.md): For Level Zero and Intel GPU support.
- [**OpenCL**](install-ocl.md): For generic OpenCL / SPIR-V devices.
- [**Apple Metal**](install-metal.md): For experimental macOS GPU support.

---

## Advanced Installation Methods

For complex scenarios, such as linking AdaptiveCpp directly into LLVM or performing a 2-stage bootstrap build on macOS, see the [**Advanced Installation Guide**](advanced-builds.md).
