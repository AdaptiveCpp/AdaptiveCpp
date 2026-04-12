# Level Zero & SPIR-V Installation Guide

AdaptiveCpp supports SPIR-V devices through the Intel Level Zero API. This is the recommended path for Intel GPUs on Linux.

## Requirements

- **Level Zero Loader**: Installed from your distribution or Intel's repositories.
- **Level Zero Driver**: E.g., the [Intel Compute Runtime](https://github.com/intel/compute-runtime).
- **Hardware**: Primarily Intel GPUs (integrated and discrete).

> [!TIP]
> Targeting SPIR-V devices through **OpenCL** is currently more mature and may yield better results for certain devices. See the [OpenCL Guide](install-ocl.md).

---

## CMake Configuration

Pass these variables during the AdaptiveCpp build to configure Level Zero support:

| Variable | Description |
| :--- | :--- |
| `-DWITH_LEVEL_ZERO_BACKEND=ON` | Force enable the Level Zero backend. |

### Verification
After installation, verify that AdaptiveCpp can see your Level Zero devices:
```bash
acpp-info
```



