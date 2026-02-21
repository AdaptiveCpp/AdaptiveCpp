# AdaptiveCpp installation instructions for Metal (macOS)

The Metal backend is experimental. It is under active development, and not all SYCL features are supported yet. Expect rough edges.

The Metal backend allows running SYCL kernels on Apple GPUs using Apple's Metal GPU API. It is part of the generic SSCP compilation flow: AdaptiveCpp compiles SYCL kernels to LLVM IR at compile time, then translates that IR to Metal Shading Language (MSL) at runtime before submitting the kernel to the GPU.

## Requirements

* macOS 26. The backend has only been tested on macOS 26; other versions are unlikely to work.
* An Apple Silicon Mac. The backend has only been tested on Apple Silicon (M-series). Behavior on Intel-based Macs is unknown.
* Xcode or the Xcode Command Line Tools (provides the Metal framework and `xcrun`)
* The **metal-cpp** header-only wrappers from Apple (see below)
* An LLVM installation (release >= 15) with the AArch64 target enabled, as described in [the LLVM installation instructions](install-llvm.md). A [2-stage build](installing.md#using-a-2-stage-build-mac) is recommended to get a fully working generic SSCP compiler on Apple Silicon.

## Installing metal-cpp

AdaptiveCpp uses [metal-cpp](https://developer.apple.com/metal/cpp/), Apple's C++ wrapper for the Metal API. You need to download it separately.

Download the latest release from Apple's developer site and unpack it. Then point AdaptiveCpp to the directory with `-DMETAL_INCLUDE_DIR`:

```bash
-DMETAL_INCLUDE_DIR="/path/to/metal-cpp"
```

## Enabling the Metal backend

When configuring AdaptiveCpp with CMake, pass `-DWITH_METAL_BACKEND=ON` together with `-DMETAL_INCLUDE_DIR`:

```bash
cmake \
  -DCMAKE_INSTALL_PREFIX=/your/install/path \
  -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm \
  -DWITH_METAL_BACKEND=ON \
  -DMETAL_INCLUDE_DIR="/path/to/metal-cpp" \
  ..
make install
```

## Using the Metal backend

The Metal backend is part of the `generic` compilation flow, so you use it the same way as any other `generic` target:

```bash
acpp -o my_program my_program.cpp
./my_program
```

At runtime, AdaptiveCpp will automatically detect the Metal GPU and dispatch kernels to it. No additional flags are needed.

## Known limitations

The Metal backend is experimental and has the following important limitations:

* **Only SYCL is supported.** The Metal backend supports SYCL kernels only. The portable CUDA dialect (PCUDA) is not supported on Metal.

* **No full USM pointer semantics.** Metal does not currently support arbitrary GPU-side pointer dereferencing across separate allocations. All buffers that a kernel needs must be passed explicitly as kernel arguments. Passing a struct that contains a pointer to another GPU buffer — for example:

    ```cpp
    struct Entity { double* data; };
    // ... array of Entity passed to kernel ...
    ```

    and then dereferencing `entity.data` inside the kernel will cause the program to crash, because the nested pointer is not valid from the GPU's perspective. Only flat buffers passed directly as kernel arguments are safe to access.

    Full USM semantics are planned for a future release, making use of features introduced in Metal 4.

* **`double` is not supported.** Apple Silicon GPUs do not have hardware support for double-precision floating point. Support for `double` is planned for a future release as a software emulation (soft-double) for compatibility, but it will not deliver hardware-native performance.

* **64-bit atomics (`atomic64`) are not supported.** Metal does not provide 64-bit atomic operations on Apple Silicon GPUs.

* **SYCL event performance.** The current implementation of SYCL events is not optimal from a performance standpoint. This is a known issue that will be addressed in a future release.

* **`sycl::stream` / printf** is not supported.

## Checking that the backend is active

After installation, you can verify that AdaptiveCpp can see your Metal device:

```bash
acpp-info
```

You should see a Metal device listed in the output.
