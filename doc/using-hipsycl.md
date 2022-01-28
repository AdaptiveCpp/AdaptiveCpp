# Using hipSYCL in projects
It is recommended to use the CMake integration for larger projects.

## Compiling with syclcc
`syclcc` is the compilation driver used by hipSYCL to build the final compiler invocations.
After installing hipSYCL, it can be used as a standalone tool to manually build source files similarly to regular compilers, or it can be integrated in build systems other than CMake.
For example, compiling a SYCL source `example.cpp` to an executable, while targeting CPU and CUDA backends, is possible using `syclcc -o example example.cpp -O3 --hipsycl-targets="omp;cuda:sm_61"`.

The full excerpt from `syclcc --help` follows below. Note the options can also be set via environment variables or corresponding CMake options. Default values can be set in `/hipsycl/install/path/etc/hipSYCL/syclcc.json`.
```
syclcc [hipSYCL compilation driver], Copyright (C) 2018-2022 Aksel Alpay
Usage: syclcc <options>

Options are:
--hipsycl-platform=<value>
  [can also be set with environment variable: HIPSYCL_PLATFORM=<value>]
  [default value provided by field 'default-platform' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  (deprecated) The platform that hipSYCL should target. Valid values:
    * cuda: Target NVIDIA CUDA GPUs
    * rocm: Target AMD GPUs running on the ROCm platform
    * cpu: Target only CPUs

--hipsycl-clang=<value>
  [can also be set with environment variable: HIPSYCL_CLANG=<value>]
  [default value provided by field 'default-clang' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the clang executable that should be used for compilation
    (Note: *must* be compatible with the clang version that the 
     hipSYCL clang plugin was compiled against!)

--hipsycl-cuda-path=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_PATH=<value>]
  [default value provided by field 'default-cuda-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the CUDA toolkit installation directry

--hipsycl-rocm-path=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_PATH=<value>]
  [default value provided by field 'default-rocm-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to the ROCm installation directory

--hipsycl-gpu-arch=<value>
  [can also be set with environment variable: HIPSYCL_GPU_ARCH=<value>]
  [default value provided by field 'default-gpu-arch' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  (deprecated) The GPU architecture that should be targeted when compiling for GPUs.
    For CUDA, the architecture has the form sm_XX, e.g. sm_60 for Pascal.
    For ROCm, the architecture has the form gfxYYY, e.g. gfx900 for Vega 10, gfx906 for Vega 20.

--hipsycl-cpu-cxx=<value>
  [can also be set with environment variable: HIPSYCL_CPU_CXX=<value>]
  [default value provided by field 'default-cpu-cxx' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The compiler that should be used when targeting only CPUs.

--hipsycl-clang-include-path=<value>
  [can also be set with environment variable: HIPSYCL_CLANG_INCLUDE_PATH=<value>]
  [default value provided by field 'default-clang-include-path' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  The path to clang's internal include headers. Typically of the form $PREFIX/include/clang/<version>/include. Only required by ROCm.

--hipsycl-squential-link-line=<value>
  [can also be set with environment variable: HIPSYCL_SEQUENTIAL_LINK_LINE=<value>]
  [default value provided by field 'default-sequential-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the sequential backend

--hipsycl-squential-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_SEQUENTIAL_CXX_FLAGS=<value>]
  [default value provided by field 'default-sequential-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the sequential backend

--hipsycl-omp-link-line=<value>
  [can also be set with environment variable: HIPSYCL_OMP_LINK_LINE=<value>]
  [default value provided by field 'default-omp-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the OpenMP backend.

--hipsycl-omp-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_OMP_CXX_FLAGS=<value>]
  [default value provided by field 'default-omp-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the OpenMP backend

--hipsycl-rocm-link-line=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_LINK_LINE=<value>]
  [default value provided by field 'default-rocm-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the ROCm backend.

--hipsycl-rocm-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_ROCM_CXX_FLAGS=<value>]
  [default value provided by field 'default-rocm-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the ROCm backend

--hipsycl-cuda-link-line=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_LINK_LINE=<value>]
  [default value provided by field 'default-cuda-link-line' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the linker for the CUDA backend.

--hipsycl-cuda-cxx-flags=<value>
  [can also be set with environment variable: HIPSYCL_CUDA_CXX_FLAGS=<value>]
  [default value provided by field 'default-cuda-cxx-flags' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
 The arguments passed to the compiler to compile for the CUDA backend

--hipsycl-config-file=<value>
  [can also be set with environment variable: HIPSYCL_CONFIG_FILE=<value>]
  [default value provided by field 'default-config-file' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  Select an alternative path for the config file containing the default hipSYCL settings.
    It is normally not necessary for the user to change this setting. 

--hipsycl-targets=<value>
  [can also be set with environment variable: HIPSYCL_TARGETS=<value>]
  [default value provided by field 'default-targets' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  Specify backends and targets to compile for. Example: --hipsycl-targets='omp;hip:gfx900,gfx906'
    Available backends:
      * omp - OpenMP CPU backend
      * cuda - CUDA backend 
               Requires specification of targets of the form sm_XY,
               e.g. sm_70 for Volta, sm_60 for Pascal
      * cuda-nvcxx - CUDA backend with nvc++. Target specification is optional;
               if given requires the format ccXY.
      * hip  - HIP backend
               Requires specification of targets of the form gfxXYZ,
               e.g. gfx906 for Vega 20, gfx900 for Vega 10
      * spirv - use clang SYCL driver to generate spirv

--hipsycl-use-accelerated-cpu
  [can also be set by setting environment variable HIPSYCL_USE_ACCELERATED_CPU to any value other than false|off|0 ]
  [default value provided by field 'default-use-accelerated-cpu' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: Not Set]
  If set, Clang is used for host compilation and explicit compiler support
  is enabled for accelerating the nd-range parallel_for on CPU.
  Uses continuation-based synchronization to execute all work-items
  of a work-group in a single thread, eliminating scheduling overhead
  and enabling enhanced vectorization opportunities compared to the fiber variant.

--hipsycl-dryrun
  [can also be set by setting environment variable HIPSYCL_DRYRUN to any value other than false|off|0 ]
  [default value provided by field 'default-is-dryrun' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, only shows compilation commands that would be executed, 
  but does not actually execute it. 

--hipsycl-explicit-multipass
  [can also be set by setting environment variable HIPSYCL_EXPLICIT_MULTIPASS to any value other than false|off|0 ]
  [default value provided by field 'default-is-explicit-multipass' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, executes device passes as separate compiler invocation and lets hipSYCL control embedding device
  images into the host binary. This allows targeting multiple backends simultaneously that might otherwise be
  incompatible. In this mode, source code level interoperability may not be supported in the host pass.
  For example, you cannot use the CUDA kernel launch syntax[i.e. kernel <<< ... >>> (...)] in this mode. 

--hipsycl-save-temps
  [can also be set by setting environment variable HIPSYCL_SAVE_TEMPS to any value other than false|off|0 ]
  [default value provided by field 'default-save-temps' in /install/path/etc/hipSYCL/syclcc.json.]
  [current value: NOT SET]
  If set, do not delete temporary files created during compilation.


Any other options will be forwarded to the compiler.

Note: Command line arguments take precedence over environment variables.
```

## Using the CMake integration
Setting up a project using the hipSYCL CMake integration is quite straight forward.
The main points are adding `find_package(hipSYCL REQUIRED)` and after defining the targets to build, adding `add_sycl_to_target(TARGET <target_name>)` to have the compilation handled by the hipSYCL toolchain.
See the [example cmake project](../examples/CMakeLists.txt).

A typical configure command line looks like this: `cmake .. -DhipSYCL_DIR=</hipsycl/install/lib/cmake/hipSYCL> -DHIPSYCL_TARGETS="<targets>"`.
`HIPSYCL_TARGETS` has to be set either as environment variable or on the command line for the `find_package` call to succeed. See the documentation of this flag above.
