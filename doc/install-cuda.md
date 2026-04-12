# NVIDIA CUDA Installation Guide

AdaptiveCpp supports NVIDIA GPUs through two primary paths: using **Clang** (recommended for performance and generic flows) or **NVC++** (from the NVIDIA HPC SDK).

## Requirements

- **CUDA Toolkit**: Version 10.0 or later.
- **Hardware**: Any NVIDIA GPU supported by your chosen CUDA version.

---

## Path 1: Using Clang (Recommended)

Clang usually produces CUDA programs with very competitive performance compared to `nvcc` or `nvc++`.

> [!NOTE]
> For more information on compiling CUDA with Clang, please read the [LLVM CUDA documentation](http://llvm.org/docs/CompileCudaWithLLVM.html).

### Troubleshooting Version Warnings
If you use a very recent CUDA version, Clang might issue a warning that it does not yet officially support that version and will treat it as an older one. This warning can usually be safely ignored.

---

## Path 2: Using NVC++ (NVIDIA HPC SDK)

This path allows you to use `nvc++` as the underlying compiler. It is particularly useful if you want to avoid a full LLVM installation.

- **Requirement**: Install the latest release of the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).
- **Tip**: You can use the CUDA toolkit bundled with the HPC SDK.

---

## CMake Configuration

Pass these variables during the AdaptiveCpp build to configure CUDA support:

| Variable | Description |
| :--- | :--- |
| `-DWITH_CUDA_BACKEND=ON` | Force enable the CUDA backend. |
| `-DCUDA_TOOLKIT_ROOT_DIR` | Path to the CUDA installation (e.g., `/usr/local/cuda`). |
| `-DNVCXX_COMPILER` | Path to the `nvc++` executable (only if using Path 2). |
| `-DWITH_CUDA_NVCXX_ONLY=ON` | Enable if you want to use `nvc++` exclusively without LLVM/Clang. |

### Verification
After installation, verify that AdaptiveCpp can see your NVIDIA GPU:
```bash
acpp-info
```
