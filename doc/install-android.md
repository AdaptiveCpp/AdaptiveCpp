# AdaptiveCpp Build and Install Instructions for Android

Building and installing the project for Android is a complicated process, as it involves cross compilation through the Android Native Development Kit (NDK), and
installation through the Android Debug Bridge (ADB). The SYCL programs must also be cross compiled when built using a separate native AdaptiveCpp compiled
and then executed through ADB. The `generic` compilation flow is supported and enables the OpenMP (host CPU) and Vulkan backends. This does however require
push a large amount of binary dependencies to the device to enable runtime compilation, which may be limiting in a real world mobile use case.

This cross compilation process has only been tested from an Ubuntu host OS and assumes that you can access your device using ABD to
get the following information about CPU and Android version. These are used to set the `ANDROID_ABI` and `ANDROID_PLATFORM` defines
in Cmake configuration and can be replaced with the values from your own device.

```sh
$ adb shell getprop ro.product.cpu.abi
arm64-v8a
$ adb shell getprop ro.build.version.sdk
34
```

Instruction Steps:

1. [Acquire the Android NDK](#acquire-android-ndk)
1. [Cross compile clspv from source](#cross-compile-clspv) (Vulkan backend build only)
1. [Cross compile SPIR-V tools from source](#cross-compile-spirv-tools) (Vulkan backend build only)
1. [Clone VulkanHpp source](#clone-vulkanhpp) (Vulkan backend build only)
1. [Cross compile LLVM from source](#cross-compile-llvm)
1. [Cross compile AdaptiveCpp](#cross-compile-adaptivecpp)
1. [Native compile AdaptiveCpp](#native-compile-adaptivecpp)
1. [Cross compile application](#cross-compile-application)
1. [Push binaries to device and run](#push-binaries-and-run)

Each step exports the paths to artifacts that will be needed in later steps as an environment variable. If only
a OpenMP backend is desired then then Vulkan only steps may be skipped.

## Acquire Android NDK

The [Android NDK toolchain](https://developer.android.com/ndk/downloads) is the supported way to cross compile
C/C++ programs for Android. These instructions in this doc are written using the r27d LTS release.

```sh
$ wget https://dl.google.com/android/repository/android-ndk-r27d-linux.zip
$ unzip android-ndk-r27d-linux.zip
$ export NDK=$PWD/android-ndk-r27d
```

## Cross Compile clspv

> This step is need for the Vulkan backend only.

The Vulkan backend requires a clspv executable to compile LLVM IR to Vulkan SPIR-V at runtime. As a result a clspv binary
cross compiled for Android is required. This itself is a two stage process as it first requires `libclc` to be natively
compiled and then used in the cross compiled build. See the [libclc doc](https://github.com/llvm/llvm-project/tree/main/libclc#configure-for-vulkan-clspv-targets)
for details on the Vulkan specific arguments. At the time of writing clspv is using upstream clang with a version number 23 but this
is not required to match the LLVM versions used to build AdaptiveCpp, as clspv is statically compiled as an standalone binary.


```sh
$ git clone https://github.com/google/clspv.git
$ cd clspv
$ python3 utils/fetch_sources.py
$ mkdir build-clc && cd build-clc
$ cmake  ../third_party/llvm/llvm  -G Ninja \
            -DCMAKE_BUILD_TYPE=Release \
            -DLLVM_ENABLE_PROJECTS=clang \
            -DRUNTIMES_spirv64-unknown-vulkan_LLVM_ENABLE_RUNTIMES=libclc \
            -DRUNTIMES_spirv32-unknown-vulkan_LLVM_ENABLE_RUNTIMES=libclc \
            -DLLVM_RUNTIME_TARGETS="spirv64-unknown-vulkan;spirv32-unknown-vulkan;" \
            -DLLVM_TARGETS_TO_BUILD=Native
$ ninja libclc

$ cd ../ && mkdir build-ndk && cd build-ndk
$ cmake .. -GNinja \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-34 \
    -DCLSPV_EXTERNAL_LIBCLC_DIR=$PWD/../build-libclc/lib/clang/23/lib
$ ninja
$ export CLSPV_NDK_PATH=$PWD/bin/clspv
```

## Cross Compile SPIRV Tools

> This step is need for the Vulkan backend only.

The Android NDK does not provide the SPIRV Tools headers and libraries that are distributed by the
Vulkan SDK. When cross compiling the project therefore need to be manually cross compiled with the NDK
toolchain.

```sh
$ git clone https://github.com/KhronosGroup/SPIRV-Tools.git
$ cd SPIRV-Tools
$ python3 utils/git-sync-deps

$ mkdir build-ndk && cd build-ndk
$ cmake ../ -GNinja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=34 \
  -DCMAKE_INSTALL_PREFIX=$PWD/install
$ ninja install
$ export SPIRV_TOOLS_NDK_PATH=$PWD/install
$ export SPIRV_TOOLS_SOURCE_PATH=$PWD/..
```

## Clone VulkanHpp

> This step is need for the Vulkan backend only.

The Android NDK only provides the Vulkan C headers, not the layered C++ bindings provided by Khronos
which are part of the Vulkan SDK and used in the AdaptiveCpp source code. These source of these
headers needs to be clone but not built.

```sh
$ git clone --recurse-submodules https://github.com/KhronosGroup/Vulkan-Hpp.git
$ export VULKAN_HPP_PATH=$PWD/Vulkan-Hpp
```

## Cross Compile LLVM

In order to perform SSCP compilation at runtime AdaptiveCpp needs an cross compiled LLVM and tools to link against. We [build LLVM](install-llvm.md)
from source separately as a step prior to cross compiling AdaptiveCpp. Here we only use "Aarch64" for the LLVM targets to build as
we know that's the CPU architecture of the Android device we're cross compiling for.

These instructions build LLVM version 20, and this should then match the LLVM version used to build a host AdaptiveCpp in
a later step.

```sh
$ git clone https://github.com/llvm/llvm-project -b release/20.x
$ cd llvm-project
$ mkdir build-ndk && cd build-ndk

$ cmake ../llvm -GNinja
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DLLVM_ENABLE_PROJECTS="clang;lld;openmp" \
  -DLLVM_ENABLE_RUNTIMES=compiler-rt \
  -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
  -DLLVM_ENABLE_ASSERTIONS=OFF \
  -DLLVM_TARGETS_TO_BUILD="AArch64" \
  -DLLVM_INCLUDE_BENCHMARKS=0 \
  -DLLVM_INCLUDE_EXAMPLES=0 \
  -DLLVM_INCLUDE_TESTS=0 \
  -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
  -DCMAKE_INSTALL_RPATH=$PWD/install/lib \
  -DLLVM_ENABLE_OCAMLDOC=OFF \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
  -DLLVM_BUILD_LLVM_DYLIB=ON

$ ninja install
$ export LLVM_NDK_INSTALL=$PWD/install
```

> Note: The non-essential component `llvm/tools/bugpoint-passes` may need to be disabled in the build due to compilation issues,
> this can be done by putting an early `return()` in the `CMakeLists.txt`

## Cross Compile AdaptiveCpp

Now that we have all the cross compilation dependencies prepared we can can cross compile AdaptiveCpp itself. We need to
pass some extra arguments to native LLVM tools that match the LLVM version being linked in the cross compile. This
allows the `libkernel` builtins to be created.

```sh
$ cd <acpp root>
$ mkdir build-ndk && cd build-ndk

$ cmake .. -GNinja \
  -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-34 \
  -DCMAKE_INSTALL_PREFIX=$PWD/install \
  -DLLVM_DIR=$LLVM_NDK_INSTALL/lib/cmake/llvm \
  -DCLANG_EXECUTABLE_PATH=/usr/bin/clang-20 \
  -DACPP_LLVMLINK_PATH=/usr/bin/llvm-link-20 \
  -DTARGET_TRIPLE=aarch64-unknown-linux-android34

$ ninja install
$ export ACPP_NDK_INSTALL=$PWD/install
```

When compiling with the Vulkan backend enabled add the additional following arguments:

* `-DWITH_VULKAN_BACKEND=ON`
* `-DCMAKE_PROGRAM_PATH=$CLSPV_NDK_PATH`
* `-DSPIRV-Tools_DIR=$SPIRV_TOOLS_NDK_PATH/lib/cmake/SPIRV-Tools`
* `-DACPP_VULKAN_HPP_PATH=$VULKAN_HPP_PATH`
* `-DACPP_SPIRV_HEADER_PATH=$SPIRV_TOOLS_SOURCE_PATH/external/spirv-headers`

## Native Compile AdaptiveCpp

Now we have cross compiled AdaptiveCpp libraries to use on our device, we also need to be able
to cross compile SYCL application. To do that on the host a native AdaptiveCpp compiler is required, which can
be built using the [standard build and install instructions](installing.md). To build a native AdaptiveCpp from the same source,
using an LLVM from the same version, and with the same backends enabled.

The commands to to this are omitted from these instructions, but so we can refer to the native build in later instructions,
lets export the path to the install directory

```sh
$ export ACPP_NATIVE_INSTALL=<path/to/native/acpp/build/install>
```

## Cross Compile Application

Finally we have all the tools now to build and run an application. For illustration purposes, this guide
will compile and run the follow simple application.

```cpp
// android_sycl_test.cpp
#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  sycl::device d{sycl::default_selector{}};
  sycl::queue q(d, sycl::property::queue::in_order());

  std::string device = d.get_info<sycl::info::device::name>();
  std::cout << "Default-selected queue runs on device: " << device << std::endl;

  constexpr size_t N = 1024;
  int *devicePtr = sycl::malloc_device<int>(N, q);

  q.parallel_for(N, [=](sycl::id<1> idx) {
    devicePtr[idx] = idx;
  });

  std::vector<int> dataHost(N);
  q.copy(devicePtr, dataHost.data(), N).wait();

  bool success = true;
  for (int i = 0; i < N; i++) {
    success = success && (dataHost[i] == i);
  }

  if (success) {
    std::cout << "SYCL application SUCCESS" << std::endl;
  } else {
    std::cout << "SYCL application FAILED" << std::endl;
  }

  sycl::free(devicePtr, q);

  return 0;
}
```

To do the compilation itself we use the AdaptiveCpp compiler for the SYCL frontend,
but point to the sysroot resources for the NDK.

```sh
$ $ACPP_NATIVE_INSTALL/bin/acpp android_sycl_test.cpp -o android_sycl_test  \
                                --target=aarch64-linux-android34 \
                                --sysroot=$NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot \
                                -static-libstdc++ \
                                --rtlib=compiler-rt \
                                -resource-dir=$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/18/ \
                                -L $ACPP_NDK_INSTALL/lib
```

## Push Binaries and Run

### Push Binaries

We now have our cross compiled application and its dependencies, but we still need a way to get them onto
the Android device and run them. To do this we use ADB with the `shell` and `push` commands, `shell`
runs an interactive shell and the first thing we do is create a directory to work in called `sycl` under
`/data/local/tmp`.

```sh
$ adb shell
(adb) $ cd /data/local/tmp/
(adb) $ mkdir sycl
```

We can then push our binaries there using the following command

```sh
$ adb push <file> /data/local/tmp/sycl
```

For the cross compiled AdaptiveCpp build we compress the whole `install` to push, and then extract on device.

```sh
$ tar -cvf acpp_ndk_install.tar $ACPP_NDK_INSTALL

$ adb shell
(adb) $ cd /data/local/tmp/sycl
(adb) $ tar -xvf acpp_ndk_install.tar
(adb) $ mv install acpp_install
```

There are lots of binaries to push (around 5GB in total!):

* `libclang-cpp.so` - In `lib` folder of build from [cross compile LLVM from source](#cross-compile-llvm) step.
* `libLLVM.so` - In `lib` folder of build from [cross compile LLVM from source](#cross-compile-llvm) step.
* `libomp.so` - From Android NDK at `$NDK/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/<version>/lib/linux/<arch>/libomp.so`.
* `opt` - In `bin` folder of build from [cross compile LLVM from source](#cross-compile-llvm) step.
* `llc` - In `bin` folder of build from [cross compile LLVM from source](#cross-compile-llvm) step.
* `ld.lld` - In `bin` folder of build from [cross compile LLVM from source](#cross-compile-llvm) step.
* `acpp_ndk_install.tar` - Output of [cross compile AdaptiveCpp](#cross-compile-adaptivecpp) step.
* `clspv` (Vulkan only) - Output of [cross compile clspv from source](#cross-compile-clspv) step.
* `android_sycl_test` - Output of [cross compile application](#cross-compile-application) step.

### Run Application

We're almost there, now we have all the binaries on device and we just need to run them. However, this is also
not straightforward as many of the default paths AdaptiveCpp tries to use at runtime are taken from CMake configuration
on the host machine and need to be overwritten with environment variables at runtime:

```sh
$ adb shell
(adb) $ cd /data/local/tmp/sycl
(adb) $ export ACPP_APPDB_DIR=$PWD/.acpp_cache
(adb) $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD:$PWD/acpp_install/lib:$PWD/acpp_install/lib/hipSYCL:$PWD/acpp_install/lib/hipSYCL/llvm-to-backend
(adb) $ export ACPP_LLC_PATH=$PWD/llc
(adb) $ export ACPP_LLD_PATH=$PWD/ld.lld
(adb) $ export ACPP_OPT_PATH=$PWD/opt
(adb) $ export ACPP_CLSPV_PATH=$PWD/clspv
```

Now everything is in place to execute the application. The previous step didn't recommend pushing the cross compiled `acpp-info` binary, but if you
do then you should see something like this:

```sh
(abd) ./acpp-info  -l
=================Backend information===================
Loaded backend 0: OpenMP
  Found device: AdaptiveCpp OpenMP host device
Loaded backend 1: Vulkan
  Found device: Adreno (TM) 750
```

Using the `ACPP_VISIBILITY_MASK` [environment_variable](env_variables.md) we can then pick between both the
Arm v8 CPU OpenMP device and Vulkan GPU device to verify that they both well.

```sh
(adb) $ ACPP_VISIBILITY_MASK=omp ./android_sycl_test
Default-selected queue runs on device: AdaptiveCpp OpenMP host device
SYCL application SUCCESS
```

On first compilation you may get a warning about mismatching target triples because the `libkernel` bitcode
was compiled on host using the host target triple, while the kernel source was compiled on device using the
target triple of the device CPU.


```sh
(adb) $ ACPP_VISIBILITY_MASK=vk ./android_sycl_test
Default-selected queue runs on device: Adreno (TM) 750
SYCL application SUCCESS
```

Congratulations is you followed these instructions all the way through to the end!
