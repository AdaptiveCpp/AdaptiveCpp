# Apple Metal Installation Guide (macOS)

> [!CAUTION]
> The Metal backend is **experimental**. It is under active development, and not all SYCL features are supported yet. Expect rough edges.

The Metal backend allows running SYCL kernels on Apple GPUs using Apple's Metal API. It is part of the generic SSCP compilation flow.

## Requirements

- **OS**: macOS 26 (Tahoe). The backend has only been tested on this version; others are unlikely to work.
- **Hardware**: Apple Silicon Mac (M-series). Intel-based Macs are not supported.
- **Tools**: Xcode or the Xcode Command Line Tools (provides `xcrun`).
- **Dependencies**:
    - **metal-cpp**: Apple's C++ wrappers (see below).
    - **LLVM**: Release >= 15 with AArch64 target enabled.

> [!TIP]
> A [2-stage build](advanced-builds.md) is highly recommended for a fully functional generic SSCP compiler on macOS.

---

## Installing metal-cpp

AdaptiveCpp uses [metal-cpp](https://developer.apple.com/metal/cpp/), Apple's C++ wrapper for the Metal API.

1.  Download the latest release from Apple's developer site.
2.  Unpack it to a directory.
3.  Point AdaptiveCpp to this directory using `-DMETAL_INCLUDE_DIR`.

---

## CMake Configuration

Pass these variables during the AdaptiveCpp build to configure Metal support:

| Variable | Description |
| :--- | :--- |
| `-DWITH_METAL_BACKEND=ON` | Force enable the Metal backend. |
| `-DMETAL_INCLUDE_DIR` | Path to the `metal-cpp` directory. |

---

## Known Limitations

- **SYCL only**: Portable CUDA (PCUDA) is not supported.
- **USM**: Full USM pointer semantics are not yet supported. All buffers must be passed explicitly as kernel arguments.
- **No `double` support**: Apple Silicon GPUs do not support double-precision hardware. Software emulation is planned.
- **Atomicity**: 64-bit atomics are not supported.
- **Streams**: `sycl::stream` and `printf` are not supported.

### Verification
After installation, verify that AdaptiveCpp can see your Apple GPU:
```bash
acpp-info
```
