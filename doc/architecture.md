# AdaptiveCpp design and architecture


AdaptiveCpp is a SYCL implementation which also supports integrating with existing heterogeneous compiler toolchains, such as CUDA, OpenMP etc. In general this means that AdaptiveCpp also needs to contain code written in other different programming models that might require specific compilers in order to be parsed (e.g. CUDA). Because of this, AdaptiveCpp is designed to interface with and rely on multiple compilers or multiple C++ dialects such as CUDA or OpenMP in its code base.

AdaptiveCpp consists of four major parts:

1. SYCL interface (consisting of SYCL host API and SYCL kernel library)
2. Runtime
3. Compiler
4. Glue

These four components are strictly separated both in terms of the directory structure and C++ namespaces:

| Component | Header files | Source files | Namespace |
|------------------|-------------------|------------------|------------------|
| SYCL Interface | `include/hipSYCL/sycl` | N/A | `hipsycl::sycl` |
| SYCL Runtime   | `include/hipSYCL/runtime` | `src/runtime` | `hipsycl::rt` |
| Compiler (clang plugin) | `include/hipSYCL/compiler` | `src/compiler` | `hipsycl::compiler` |
| Glue | `include/hipSYCL/glue` | N/A | `hipsycl::glue` |


## SYCL interface

The SYCL interface provides the SYCL classes and functions that the user actually interacts with, i.e. everything inside the `sycl::` namespace. It can be seen as divided in two parts: 
1. SYCL host API: This part of the SYCL interface is written in regular C++, only runs on the host and provides mechanisms for task submission, task management, platform and device management etc. For example `sycl::queue`, `sycl::event`, `sycl::device` belong to the host API. The SYCL host API is mainly just an interface to the SYCL runtime, which actually implements most of these features. *Backend-specific code is not allowed in the host API as a rule of thumb. Non-standard C++ code (e.g. CUDA) is absolutely not allowed in the host API.*
2. SYCL kernel library: The kernel library contains all SYCL classes and functions that are available from kernels. In general, the kernel library will make use of backend-specific functionality and may even need to be written in a backend-specific C++ dialect such as CUDA. This means that in general, a regular C++ compiler may not be able to parse the kernel library code.
  Note that there are some classes such as `sycl::accessor`, `sycl::id`, `sycl::range` that are needed both inside and outside kernel code.

Since the SYCL interface is the only part of AdaptiveCpp where backend-specific C++ dialects are allowed (in conjunction with the kernel launchers from the glue component), compiling code that relies on the SYCL interface can be substantially more complicated than regular C++ code since since special compilers may be needed.
Because of this, and to simplify the AdaptiveCpp build process, the SYCL interface is a header-only library. Consequently, the SYCL interface is only compiled during the compilation of user code when a device compiler is available.


## AdaptiveCpp runtime

The runtime implements device management, task graph management and execution, data management, backend management, scheduling and task synchronization. It interfaces with the runtime components of the supported backends (e.g. the CUDA runtime).

Unlike the SYCL interface, the runtime is not header-only - it is in fact the major component that needs to be compiled when building AdaptiveCpp. The runtime (compiled as `libacpp-rt`) is a unified library for all backends. Backends are implemented using polymorphism, i.e. as classes derived from an abstract `backend` base class. They are dynamically loaded from plugins. When the runtime is initialized, all available backend plugins are discovered and loaded. This means that, *regardless of what target the user compiles SYCL code for, all devices seen by available backends will show up when querying available devices*. 
However, in order to actually run kernels on a particular device, it is additionally necessary that the compiler component has generated code for this device. This can lead to the situation where a user can select a device for computation just fine, but then is unable to run kernels on the device if it has not been specified as compilation target when compiling the SYCL code.

Because the runtime is compiled like any regular C++ library, it *must not use functionality from the SYCL interface*, since the SYCL interface in general *cannot* be compiled by a regular C++ compiler.
In other words, while the SYCL interface calls into the runtime, the inverse is not true. The interaction between SYCL interface and runtime is one-way.

The AdaptiveCpp runtime follows a [specification](runtime-spec.md) that expands on the Khronos SYCL specification.

The following image illustrates the runtime architecture:
![Runtime architecture](/doc/img/runtime.png)

## Compiler

Compilation of user code is driven by python script called `syclcc` which provides a uniform interface to the compilers for individual backends (e.g., the CUDA compiler). When compiling for CUDA or HIP, hipSYCL additionally provides and relies on a clang plugin that augments the clang CUDA/HIP frontend to also understand SYCL. The clang plugin's main responsibility is kernel outlining, i.e. it identifies kernels and makes sure that all functions needed by the kernel are also compiled for device by attaching the CUDA/HIP `__host__ __device__` attributes.
Because of the clang plugin, clang is a dependency for the CUDA and ROCm backends. Other backends may use other compilers.

## Glue

If a connection needs to be established between some of the other components that requires intimate knowledge or implementation details of both, the `glue` part of AdaptiveCpp is the right place.

The most important part of the glue code are the *kernel launchers*. The kernel launchers, as the name suggests, implement launching a SYCL kernel for a particular backend. 
While launching a kernel is semantically part of the runtime as it is subject to the runtime's scheduling decisions, it cannot be a part of the runtime: Launching a kernel may in general require specific compilers (think e.g. of the CUDA `<<<>>>` syntax or OpenMP pragmas). At the same time, relying on special compilers is only supported in the SYCL interface.
The solution is to implement the kernel launch code as a header-only library that is included by the SYCL interface. The kernel launcher code generates a function object which, when executed, will invoke the kernel. This means we can simply pass this function object to the runtime, which will then invoke it when required without having to deal with the question of how exactly a kernel launch happens and what compiler was needed to generate the kernel.

Depending on how much of the SYCL execution model is already provided by the backend, kernel launchers can be relatively simple (e.g. on CUDA/HIP) or more complex. For example, the OpenMP CPU backend needs to also implement parts of the SYCL execution model that are not naturally available already.
