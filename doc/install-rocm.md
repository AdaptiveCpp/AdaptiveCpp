# AdaptiveCpp installation instructions for ROCm

Please install ROCm 4.0 or later as described in the ROCm readme. Make sure to also install HIP (runtime libraries and headers).

## The main compiler: Generic single-pass (--acpp-targets=generic)

AdaptiveCpp's main compiler, the generic single-pass compiler, requires that AdaptiveCpp's LLVM version is <= the LLVM version that ships with ROCm.

ROCm >= 5.3 is recommended.

## The hip interoperability compiler (--acpp-targets=hip)

When using AdaptiveCpp's hip interoperability compiler (`--acpp-targets=hip`), newer ROCm versions may require building AdaptiveCpp against newer clang versions as well. For example, ROCm 4.5 requires clang 13+.

### Troubleshooting missing device libraries

For ordinary SYCL applications, prefer the [generic single-pass compiler](compilation.md), which is the default in a typical installation:

```sh
acpp --acpp-targets=generic -O2 sourcecode.cpp
```

If you need the HIP interoperability compiler instead, specify a GPU architecture, for example `--acpp-targets=hip:gfx90a`. Replace `gfx90a` with the target architecture of your GPU; see [the target specification](using-acpp.md#requirements-for-specifying-targets-of-individual-compilation-flows).

If Clang reports `cannot find ROCm device library` with this flow, check the location of the ROCm device-library bitcode files. In a standard installation, AdaptiveCpp passes `--rocm-device-lib-path=$ACPP_ROCM_PATH/amdgcn/bitcode` to Clang. Setting `-DROCM_PATH` when configuring AdaptiveCpp does not locate a separately installed device-library package, such as one provided by Spack.

For a split installation, pass the directory containing `ockl.bc` and the other ROCm device-library bitcode files when compiling the application:

```sh
acpp --acpp-targets=hip:gfx90a \
  --rocm-device-lib-path=/path/to/rocm-device-libs/amdgcn/bitcode \
  -O2 sourcecode.cpp
```

Use the actual bitcode directory from your installation, not the HIP headers or runtime-library directory. This option is for the HIP interoperability flow; it is not needed to select an AMD GPU with the generic compiler.

## Using AdaptiveCpp with LLVM bundled from ROCm (not recommended)

Instead of building AdaptiveCpp against a regular clang/LLVM, it is also possible (but not necessarily recommended, see below) to build AdaptiveCpp against the clang/LLVM that ships with ROCm. This can be interesting if other available clang/LLVM installations are not new enough to work with the ROCm installation.

* **Such configurations may work, but are generally less tested.**
* Also note that the LLVM distributions shipping with ROCm are not official LLVM releases, and depending on when the upstream development was last merged, may have slightly diverging functionality. There are multiple known cases where this causes problems: 
  * the clang 14 from ROCm 5.0 lacks functionality that is present in official clang 14 releases. You can work around those issues by setting `-DACPP_COMPILER_FEATURE_PROFILE=minimal` at the expense of missing features, such as reduced kernel performance on CPUs and lack of [SSCP](compilation.md) support.
* **Due to the potential loss of key AdaptiveCpp functionality such as the generic SSCP compiler, these configurations are typically not recommended.**

*Note: AdaptiveCpp is by default configured to utilize the ROCm compilation flags that apply for recent clang and ROCm versions. If you are using an older clang (<= 10) or ROCm < 4, you might have to adjust `-DROCM_CXX_FLAGS` (not recommended!).*

## CMake variables:

* `-DROCM_PATH=/path/to/rocm` (default: /opt/rocm)
* `-DWITH_ROCM_BACKEND=ON` if AdaptiveCpp does not automatically enable the ROCm backend 

