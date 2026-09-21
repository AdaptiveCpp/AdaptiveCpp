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

## Interoperability with Metal

The Metal backend exposes native objects, allowing SYCL code to use Metal directly. This interop API is available only in builds with Metal support. Check for the `SYCL_EXT_ACPP_BACKEND_METAL` macro.

| What | How | Native type |
|---|---|---|
| Device | `sycl::get_native<sycl::backend::metal>(device)` | `MTL::Device*` |
| Event | `sycl::get_native<sycl::backend::metal>(event)` | `hipsycl::rt::metal_event_handle`: an `MTL::SharedEvent*` and its signal value |
| USM pointer | `sycl::get_native_allocation<sycl::backend::metal>(ptr, context)` | `MTL::Buffer*` and the pointer's offset within it |
| Queue | `h.get_native_queue<sycl::backend::metal>()` inside a custom operation | `MTL::CommandQueue*` |
| Import event | `sycl::make_event<sycl::backend::metal>(handle, context)` | `hipsycl::rt::metal_event_handle` |

`sycl::get_native<sycl::backend::metal>(queue)` is *not* provided because AdaptiveCpp may use a different queue for each operation. The command queue is available only through a custom operation's `interop_handle`. See the [custom operation documentation](enqueue-custom-operation.md).

`get_native_allocation()` accepts any pointer within a USM allocation, not just its base pointer. It returns the underlying `MTL::Buffer` and the pointer's offset within it.

```c++
#include <sycl/sycl.hpp>

sycl::queue q;
int *data = sycl::malloc_device<int>(1024, q);

q.submit([&](sycl::handler &cgh) {
  cgh.AdaptiveCpp_enqueue_custom_operation([=](sycl::interop_handle &h) {
    MTL::CommandQueue *queue = h.get_native_queue<sycl::backend::metal>();
    auto allocation = sycl::get_native_allocation<sycl::backend::metal>(
        data + 16, q.get_context());
    // allocation.buffer is the MTL::Buffer; allocation.offset is the offset
    // of data + 16 within it. Submit your own command buffer on queue here.
  });
});
q.wait();
```

`sycl::get_native()` exports a SYCL event. This allows Metal command buffers to wait for SYCL work. The returned handle contains the queue's shared event and the operation's signal value. AdaptiveCpp submits the operation before returning the handle:

```c++
sycl::event e = q.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> i) { data[i] = i[0]; });

auto handle = sycl::get_native<sycl::backend::metal>(e);
// handle.event is the MTL::SharedEvent; handle.value is its signal value
command_buffer->encodeWait(handle.event, handle.value);
```

`sycl::make_event()` imports work submitted outside SYCL. If the context has no Metal device, it throws a `sycl::exception` with `errc::backend_mismatch`. The imported event can be used like any other SYCL event:

```c++
sycl::event imported = sycl::make_event<sycl::backend::metal>(
    {shared_event, signal_value}, q.get_context());

q.submit([&](sycl::handler &cgh) {
  cgh.depends_on(imported);
  cgh.parallel_for(sycl::range<1>{1024}, [=](sycl::id<1> i) { data[i] += 1; });
});
```

## Known limitations

The Metal backend is experimental and has the following important limitations:

* **`double` is not supported.** Apple Silicon GPUs do not have hardware support for double-precision floating point. Support for `double` is planned for a future release as a software emulation (soft-double) for compatibility, but it will not deliver hardware-native performance.

* **64-bit atomics (`atomic64`) are not supported.** Metal does not provide 64-bit atomic operations on Apple Silicon GPUs.

* **SYCL event performance.** Every SYCL event must be signalled by a Metal command buffer. An operation that returns a regular event is therefore committed separately instead of being batched with adjacent operations. If you do not need per-operation events, the [coarse-grained events extension](extensions.md#acpp_ext_coarse_grained_events) avoids this overhead and noticeably reduces launch latency.

* **`sycl::stream` / printf** is not supported.

## Checking that the backend is active

After installation, you can verify that AdaptiveCpp can see your Metal device:

```bash
acpp-info
```

You should see a Metal device listed in the output.
