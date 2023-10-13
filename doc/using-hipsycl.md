# Using AdaptiveCpp in projects
It is recommended to use the CMake integration for larger projects. See the section on the cmake integration below. Alternatively, `acpp` can be used directly as a compiler.

## AdaptiveCpp targets specification

Both `acpp` and the cmake integration expect an AdaptiveCpp targets specification. This specification defines which compilation flows AdaptiveCpp should enable, and which devices from a compilation flow AdaptiveCpp should target during compilation. In general, it has the form:

```
"flow1:target1,target2,...;flow2:...;..."
```
and can be passed either as `acpp` command line argument, environment variable or CMake argument depending on whether `acpp` or `cmake` is used.

"compilation flow" refers to one of the available compilation flows defined in the [compilation flow](compilation.md) documentation.

### Requirements for specifying targets of individual compilation flows

Whether a compilation flow needs to be followed by a target list or not varies between the available flows and is described below.

For the following compilation flows, targets cannot be specified:
* `omp.*`
* `spirv`

For the following compilation flows, targets can optionally be specified:
* `cuda-nvcxx` - Targets take the format of `ccXY` where `XY` stands for the compute capability of the device.

For the following compilation flows, targets must be specified:
* `cuda.*` - The target format is defined by clang and takes the format of `sm_XY`. For example:
  * `sm_52`: NVIDIA Maxwell GPUs
  * `sm_60`: NVIDIA Pascal GPUs
  * `sm_70`: NVIDIA Volta GPUs
* `hip.*` - The target format is defined by clang and takes the format of `gfxXYZ`. For example:
  * `gfx900`: AMD Vega 10 GPUs (e.g. Radeon Vega 56, Vega 64)
  * `gfx906`: AMD Vega 20 GPUs (e.g. Radeon VII, Instinct MI50)
  * `gfx908`: AMD CDNA GPUs (e.g Instinct MI100)

### Abbreviations

For some compilation flows, abbreviations exist that will be resolved by AdaptiveCpp to one of the available compilation flows:
* `omp` will be translated 
  * into `omp.accelerated` 
     * if AdaptiveCpp has been built with support for accelerated CPU and the host compiler is the clang that AdaptiveCpp has been built with or
     * if `--acpp-use-accelerated-cpu` is set. If the accelerated CPU compilation flow is not available (e.g. AdaptiveCpp has been compiled without support for it), compilation will abort with an error.
  * into `omp.library-only` otherwise
* `cuda` will be translated
  * into `cuda.explicit-multipass`
    * if another integrated multipass has been requested, or another backend that would conflict with `cuda.integrated-multipass`. AdaptiveCpp will emit a warning in this case, since switching to explicit multipass can change interoperability guarantees (see the [compilation](compilation.md) documentation).
    * if `--acpp-explicit-multipass` is set explicitly
  * into `cuda.integrated-multipass` otherwise
* `hip` will be translated into `hip.integrated-multipass`

Of course, the desired flows can also always be specified explicitly.

### Examples

* `"omp.library-only;cuda.explicit-multipass:sm_61;sm_70"`  - compiles for the CPU backend and Pascal and Volta era GPUs
* `"omp;cuda:sm_70;hip:gfx906"`  - compiles for the CPU backend (library or accelerated), NVIDIA Volta era GPUs via explicit multipass, AMD Vega 20 GPUs
* `"omp.accelerated;cuda:sm_70;spirv`" - compiles for the CPU backend (compiler accelerated), NVIDIA Volta era GPUs, and SPIR-V devices
* `"omp;cuda-nvcxx"` - compiles for the CPU backend and NVIDIA GPUs using nvc++

### Offloading C++ standard parallelism

See [here](stdpar.md) for details on how to offload C++ standard STL algorithms using AdaptiveCpp.

## Manually compiling with acpp
`acpp` is the compilation driver used by AdaptiveCpp to build the final compiler invocations.
After installing AdaptiveCpp, it can be used as a standalone tool to manually build source files similarly to regular compilers, or it can be integrated in build systems other than CMake.
For example, compiling a SYCL source `example.cpp` to an executable, while targeting CPU and CUDA backends, is possible using `acpp -o example example.cpp -O3 --acpp-targets="omp;cuda:sm_61"`.

The full excerpt from `acpp --help` follows below. Note the options can also be set via environment variables or corresponding CMake options. Default values can be set in `/acpp/install/path/etc/hipSYCL/syclcc.json`.
```
acpp [hipSYCL compilation driver], Copyright (C) 2018-2022 Aksel Alpay and the hipSYCL project
  hipSYCL version: 0.9.2
  Installation root: /install/path
  Plugin LLVM version: <version>, can accelerate CPU: <bool>
  Available runtime backends:
     librt-backend-<name>.so
     librt-backend-<name>.so
Usage: acpp <options>

Options are:
--acpp-platform=<value>
  [can also be set with environment variable: HIPSYCL_PLATFORM=<value>]
  [default value provided by field 'default-platform' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  (deprecated) The platform that hipSYCL should target. Valid values:
    * cuda: Target NVIDIA CUDA GPUs
    * rocm: Target AMD GPUs running on the ROCm platform
    * cpu: Target only CPUs

--acpp-clang=<value>
  [can also be set with environment variable: HIPSYCL_CLANG=<value>]
  [default value provided by field 'default-clang' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the clang executable that should be used for compilation
    (Note: *must* be compatible with the clang version that the 
     hipSYCL clang plugin was compiled against!)

--acpp-nvcxx=<value>
  [can also be set with environment variable: HIPSYCL_NVCXX=<value>]
  [default value provided by field 'default-nvcxx' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the nvc++ executable that should be used for compilation
    with the cuda-nvcxx backend.

--acpp-cuda-path=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_PATH=<value>]
  [default value provided by field 'default-cuda-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the CUDA toolkit installation directry

--acpp-rocm-path=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_PATH=<value>]
  [default value provided by field 'default-rocm-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the ROCm installation directory

--acpp-gpu-arch=<value>
  [can also be set with environment variable: HIPSYCL_GPU_ARCH=<value>]
  [default value provided by field 'default-gpu-arch' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  (deprecated) The GPU architecture that should be targeted when compiling for GPUs.
    For CUDA, the architecture has the form sm_XX, e.g. sm_60 for Pascal.
    For ROCm, the architecture has the form gfxYYY, e.g. gfx900 for Vega 10, gfx906 for Vega 20.

--acpp-cpu-cxx=<value>
  [can also be set with environment variable: HIPSYCL_CPU_CXX=<value>]
  [default value provided by field 'default-cpu-cxx' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The compiler that should be used when targeting only CPUs.

--acpp-clang-include-path=<value>
  [can also be set with environment variable: HIPSYCL_CLANG_INCLUDE_PATH=<value>]
  [default value provided by field 'default-clang-include-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to clang's internal include headers. Typically of the form $PREFIX/include/clang/<version>/include. Only required by ROCm.

--acpp-squential-link-line=<value>
  [can also be set with environment variable: HIPSYCL_SEQUENTIAL_LINK_LINE=<value>]
  [default value provided by field 'default-sequential-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the sequential backend

--acpp-squential-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_SEQUENTIAL_CXX_FLAGS=<value>]
  [default value provided by field 'default-sequential-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the sequential backend

--acpp-omp-link-line=<value>
  [can also be set with environment variable: HIPSYCL_OMP_LINK_LINE=<value>]
  [default value provided by field 'default-omp-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the OpenMP backend.

--acpp-omp-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_OMP_CXX_FLAGS=<value>]
  [default value provided by field 'default-omp-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the OpenMP backend

--acpp-rocm-link-line=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_LINK_LINE=<value>]
  [default value provided by field 'default-rocm-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the ROCm backend.

--acpp-rocm-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_CXX_FLAGS=<value>]
  [default value provided by field 'default-rocm-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the ROCm backend

--acpp-cuda-link-line=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_LINK_LINE=<value>]
  [default value provided by field 'default-cuda-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the CUDA backend.

--acpp-cuda-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_CXX_FLAGS=<value>]
  [default value provided by field 'default-cuda-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the CUDA backend

--acpp-config-file=<value>
  [can also be set with environment variable: HIPSYCL_CONFIG_FILE=<value>]
  [default value provided by field 'default-config-file' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  Select an alternative path for the config file containing the default hipSYCL settings.
    It is normally not necessary for the user to change this setting. 

--acpp-targets=<value>
  [can also be set with environment variable: HIPSYCL_TARGETS=<value>]
  [default value provided by field 'default-targets' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  Specify backends and targets to compile for. Example: --acpp-targets='omp;hip:gfx900,gfx906'
    Available backends:
      * omp - OpenMP CPU backend
               Backend Flavors:
               - omp.library-only: Works with any OpenMP enabled CPU compiler.
                                   Uses Boost.Fiber for nd_range parallel_for support.
               - omp.accelerated: Uses clang as host compiler to enable compiler support
                                  for nd_range parallel_for (see --acpp-use-accelerated-cpu).
      * cuda - CUDA backend 
               Requires specification of targets of the form sm_XY,
               e.g. sm_70 for Volta, sm_60 for Pascal
               Backend Flavors:
               - cuda.explicit-multipass: CUDA backend in explicit multipass mode 
                                          (see --acpp-explicit-multipass)
               - cuda.integrated-multipass: Force CUDA backend to operate in integrated
                                           multipass mode.
      * cuda-nvcxx - CUDA backend with nvc++. Target specification is optional;
               if given requires the format ccXY.
      * hip  - HIP backend
               Requires specification of targets of the form gfxXYZ,
               e.g. gfx906 for Vega 20, gfx900 for Vega 10
      * spirv - use clang SYCL driver to generate spirv

--acpp-use-accelerated-cpu
  [can also be set by setting environment variable HIPSYCL_USE_ACCELERATED_CPU to any value other than false|off|0 ]
  [default value provided by field 'default-use-accelerated-cpu' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, Clang is used for host compilation and explicit compiler support
  is enabled for accelerating the nd-range parallel_for on CPU.
  Uses continuation-based synchronization to execute all work-items
  of a work-group in a single thread, eliminating scheduling overhead
  and enabling enhanced vectorization opportunities compared to the fiber variant.

--acpp-dryrun
  [can also be set by setting environment variable HIPSYCL_DRYRUN to any value other than false|off|0 ]
  [default value provided by field 'default-is-dryrun' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, only shows compilation commands that would be executed, 
  but does not actually execute it. 

--acpp-explicit-multipass
  [can also be set by setting environment variable HIPSYCL_EXPLICIT_MULTIPASS to any value other than false|off|0 ]
  [default value provided by field 'default-is-explicit-multipass' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, executes device passes as separate compiler invocation and lets hipSYCL control embedding device
  images into the host binary. This allows targeting multiple backends simultaneously that might otherwise be
  incompatible. In this mode, source code level interoperability may not be supported in the host pass.
  For example, you cannot use the CUDA kernel launch syntax[i.e. kernel <<< ... >>> (...)] in this mode. 

--acpp-save-temps
  [can also be set by setting environment variable HIPSYCL_SAVE_TEMPS to any value other than false|off|0 ]
  [default value provided by field 'default-save-temps' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, do not delete temporary files created during compilation.

--acpp-version
  Print AdaptiveCpp version and configuration

--help
  Print this help message


Any other options will be forwarded to the compiler.

Note: Command line arguments take precedence over environment variables.
```

## Using the CMake integration
Setting up a project using the AdaptiveCpp CMake integration is quite straight forward.
The main points are adding `find_package(AdaptiveCpp REQUIRED)` and after defining the targets to build, adding `add_sycl_to_target(TARGET <target_name>)` to have the compilation handled by the AdaptiveCpp toolchain.
See the [example cmake project](../examples/CMakeLists.txt).

A typical configure command line looks like this: `cmake .. -DAdaptiveCpp_DIR=/acpp/install/dir/lib/cmake/AdaptiveCpp -DACPP_TARGETS="<targets>"`.
`ACPP_TARGETS` has to be set either as environment variable or on the command line for the `find_package` call to succeed. See the documentation of this flag above.

If the accelerated CPU flow has been built, `-DACPP_USE_ACCELERATED_CPU=ON/OFF` can be used to override whether `omp` should refer to the `omp.library-only` or `omp.accelerated` compilation flow.
