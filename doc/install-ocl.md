# OpenCL & SPIR-V Installation Guide

AdaptiveCpp supports generic OpenCL devices that can ingest SPIR-V. This is often the most mature path for many non-NVIDIA/AMD devices.

## Requirements

- **OpenCL Implementation**: A working OpenCL driver and ICD loader.
- **SPIR-V Support**: The device must support SPIR-V ingestion and ideally the **Intel USM** (Unified Shared Memory) extension.
- **Hardware**: Various GPUs and accelerators (e.g., Intel, specialized accelerators).

---

## SPIR-V Translation

In order to generate correct SPIR-V code, AdaptiveCpp uses a custom fork of the **Khronos LLVM-SPIRV Translator**.

> [!IMPORTANT]
> AdaptiveCpp will **automatically fetch and build** the correct version of the translator for your LLVM version during the build process. You do not need to install it manually.

---

## CMake Configuration

Pass these variables during the AdaptiveCpp build to configure OpenCL support:

| Variable | Description |
| :--- | :--- |
| `-DWITH_OPENCL_BACKEND=ON` | Force enable the OpenCL backend. |
| `-DOpenCL_LIBRARY` | Path to the `libOpenCL.so` library if not detected automatically. |
| `-DOpenCL_INCLUDE_DIR` | Path to OpenCL headers if not detected. |

### Verification
After installation, verify that AdaptiveCpp can see your OpenCL devices:
```bash
acpp-info
```

