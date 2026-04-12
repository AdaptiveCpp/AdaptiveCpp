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

| Compilation Flow | Target Hardware | Requirements | Guide |
| :--- | :--- | :--- | :--- |
| `omp.library-only` | Any CPU | Any OpenMP compiler | - |
| `omp.accelerated` | Any CPU | LLVM 15-20 | [LLVM](install-llvm.md) |
| `cuda.*` | NVIDIA GPUs | CUDA 10+, LLVM 15-20 | [CUDA](install-cuda.md) |
| `cuda-nvcxx` | NVIDIA GPUs | NVIDIA HPC SDK | [CUDA](install-cuda.md) |
| `hip.*` | AMD GPUs | ROCm 4.0+, LLVM 15-20 | [ROCm](install-rocm.md) |
| `generic` | All Vendors | LLVM 15-20, Backend Runtimes | [LLVM](install-llvm.md) |
| `metal` | Apple GPUs | macOS, [metal-cpp](https://developer.apple.com/metal/cpp/) | [Metal](install-metal.md) |

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
