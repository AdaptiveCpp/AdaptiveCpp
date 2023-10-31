# AdaptiveCpp compilation model


AdaptiveCpp relies on the fact that many existing programming models as well as SYCL are single-source programming models based on C++. This means that it is possible to extend existing toolchains, such as the CUDA and the HIP toolchains, to also support SYCL code. AdaptiveCpp does that by using clang's CUDA and HIP toolchains with a custom clang plugin. Additionally, AdaptiveCpp can utilize the clang SYCL frontend to generate SPIR-V code.
AdaptiveCpp contains mechanisms to embed and aggregate compilation results from multiple toolchains into a single binary, allowing it to effectively combine multiple toolchains into one. This is illustrated here:

![acpp design](/doc/img/acpp.png)


AdaptiveCpp distinguishes and supports several compilation models:

1. Compilation flows focused on interoperability with other programming models
   1. *library-only*, where AdaptiveCpp acts as a library for a third-party compiler
   2. *SMCP* (single-source, multiple compiler pass) models, where AdaptiveCpp extends existing heterogeneous toolchains to also understand SYCL constructs. These toolchains rely on performing multiple compiler passes for host and device code. Here, we distinguish two flavors:
      1. *integrated multipass*, where host and device compilation passes are handled by clang's CUDA and HIP drivers. This mode allows for the most interoperability with backends because backend-specific language extensions are also available in the host pass. For example, kernels can be launched using the `<<<>>>` syntax. However, limitations in clang's compilation drivers also affect AdaptiveCpp in this mode. In particular, CUDA and HIP cannot be targeted simultaneously because host code cannot support language extensions from *both* at the same time.
      2. *explicit multipass*, where host and device compilation passes are handled by AdaptiveCpp's `acpp` compiler driver. Here, `acpp` invokes backend-specific device passes, which then result in a compiled kernel image for the target device. `acpp` then embeds the compiled kernel image into the host binary. At runtime, the kernel is extracted from the image and invoked using runtime functions instead of language extensions. As a consequence, explicit multipass compilation for one backend can be combined with arbitrary other backends simultaneously, allowing AdaptiveCpp to target multiple device backends at the same time. Note that in explicit multipass, different guarantees might be made regarding the availability of backend-specific language extensions in the host pass compared to integrated multipass. See the section on language extension guarantees below for more details.
2. *Generic SSCP* (Single-source, single compiler pass), where AdaptiveCpp compiles kernels to a generic representation in LLVM IR, which is then lowered at runtime to backend-specific formats such as PTX or SPIR-V as needed. Unlike the SMCP flows, there is only a single compiler invocation, and the code is only parsed once, no matter how many devices or backends are utilized. This flow can potentially provide *the most portability with the lowest compile times*, although it is still experimental work-in-progress, and support levels may vary for the different backends.
3. *Compiler-accelerated host pass*, where a regular C++ host pass is augmented with additional compiler transformations to increase performance of certain SYCL constructs when running on CPU.

Not all backends support all models. The following compilation flows are currently supported, and can be requested as backend targets in `acpp`. Note that some are available in both explicit multipass and integrated multipass flavor:

| Compilation flow | Target hardware | Model | Short description |
|------------------|-------------------|-------------------|-------------------|
| `omp.library-only` | Any CPU | Library-only | CPU backend (OpenMP library) |
| `omp.accelerated` | Any CPU supported by LLVM | Accelerated host pass | CPU backend (OpenMP library with additional clang acceleration) |
| `cuda.integrated-multipass` | NVIDIA GPUs | Integrated SMCP | CUDA backend|
| `cuda.explicit-multipass` | NVIDIA GPUs | Explicit SMCP | CUDA backend |
| `cuda-nvcxx` | NVIDIA GPUs | Library-only | CUDA backend using nvc++ |
| `hip.integrated-multipass` | AMD GPUs (supported by ROCm) | Integrated SMCP | HIP backend |
| `hip.explicit-multipass` | AMD GPUs (supported by ROCm) | Explicit SMCP | HIP backend |
| `spirv` | Intel GPUs | Explicit SMCP | SPIR-V/Level Zero backend |
| `generic` | (see below) | Generic SSCP | Generic single-pass flow |

**Note:** 
* Explicit multipass requires building AdaptiveCpp against a clang that supports `__builtin_unique_stable_name()` (available in clang 11), or clang 13 or newer as described in the [installation documentation](installing.md). `hip.explicit-multipass` requires clang 13 or newer.
* Generic SSCP requires clang 14 or newer.

## Language extension guarantees

AdaptiveCpp allows using backend-specific language extensions (e.g. CUDA/HIP C++). The precise guarantees about the availability of these extensions are as follows:

* If a backend runs on a compiler that provides a unified, single compilation pass for both host and device, backend-specific language extensions are always available. Currently this only affects the CUDA-nvc++ backend.
* If the compiler relies on separate compilation passes for host and device:
  * In device compilation passes, backend-specific language extensions are always available.
  * In host compilation passes, the following applies:
    * If the backend runs in integrated multipass mode, backend-specific language extensions are available.
    * If the backend runs in explicit multipass mode:
      * For SPIR-V, language extensions are always available
      * For CUDA and HIP: Language extensions from *one* of them are available in the host pass.
        * If one of them runs in integrated multipass and one in explicit multipass, language extensions from the one in integrated multipass are available
        * If both are in explicit multipass, `acpp` will currently automatically pick one that will have language extensions enabled in the host pass.

## Generic SSCP compilation flow

**Note:** This flow is work-in-progress and support level may vary quickly.

AdaptiveCpp supports a generic single-pass compiler flow, where a single compiler invocation generates both host and device code. The SSCP compilation consists of two stages:
1. Stage 1 happens at compile time: During the regular C++ host compilation, AdaptiveCpp extracts LLVM IR for kernels with backend-independent representations of builtins, kernel annotations etc. This LLVM IR is embedded in the host code. During stage 1, it is not yet known on which device(s) the code will ultimately run.
2. Stage 2 typically happens at runtime: The embedded device IR is passed to AdaptiveCpp's `llvm-to-backend` infrastructure, which lowers the IR to backend-specific formats, such as NVIDIA's PTX, SPIR-V or amdgcn code. Unlike stage 1, stage 2 assumes that the target device is known. While stage 2 typically happens at runtime, support for precompiling to particular devices and formats could be added in the future.

The generic SSCP design has several advantages over other implementation choices:
* There is a single code representation across all backends, which allows implementing JIT-based features such as runtime kernel fusion in a backend-independent way;
* Code is only parsed a single time, which can result is significant compilation speedups especially for template-heavy code;
* Binaries inherently run on a wide range of hardware, without the user having to precompile for particular devices, and hence making assumptions where the binary will ultimately be executed ("Compile once, run anywhere").

The generic SSCP flow can potentially provide very fast compile times, very good portability and good performance.

### Implementation status

Currently, the SSCP flow is implemented for
* CUDA devices
* SPIR-V devices through oneAPI Level Zero
* AMD ROCm devices

Some features (e.g. SYCL 2020 reductions or group algorithms) are not yet implemented.

### How it works

#### IR constants

The SSCP kernel extraction relies on the concept of what we refer to as IR constants. IR constants are global variables, that are non-const and without defined value when parsing the code, but will be turned into constants later during the processing of LLVM IR. This is a similar idea to e.g. SYCL 2020 specialization constants, and indeed specialization constants could be implemented on top of IR constants.

Stage 1 IR constants are hard-wired. The following important S1 IR constants exist:
* A string containing the device LLVM IR bitcode
* Whether the LLVM module contains the host code
* Whether the LLVM module contains the device code.

Stage 2 IR constants are intended to provide information that requires knowledge of the target device, such as backend, device, and so on. Stage 2 IR constants can also be programmatically added by the user.

After AdaptiveCpp sets the value of an IR constant, it runs constant propagation and dead code elimination passes. This causes if statements depending entirely on IR constants to be trivially optimized away - causing either removal of the code contained in the if brach, or hardwiring the code.

#### Stage 1: Kernel extraction

During stage 1, AdaptiveCpp clones the module containing the regular host IR, and sets the IR constants such that one is identified as host code, and one is identified as device code.
The kernel function calls are guarded inside the AdaptiveCpp headers by an if-statement depending on the IR constant signifying device compilation. This causes kernel code only to end up in the device module, and host code to end up only in the host module. To be sure that no host code remains in the device module, AdaptiveCpp runs additional passes in the device module to remove all code not reachable from kernel entry points.

The implementation of SYCL builtins contains an if/else branch depending on the IR constant signifying device compilation. One branch invokes the externally defined SSCP builtins following the naming scheme `__hipsycl_sscp_*`, while the other branch invokes regular host builtins.
This allows SYCL kernels to simultaneously run correctly both on the host as well as on SSCP-supported devices.

The final LLVM IR device bitcode is then embedded into a stage 1 IR constant string in the host module.

#### Stage 2: llvm-to-backend

During stage 2, the `llvm-to-backend` infrastructure is responsible for turning the generic LLVM IR into something that a backend can actually execute. This means in particular:
- Flavoring the LLVM IR such that the appropriate LLVM backend can handle the code; e.g. by correctly mapping address spaces, attaching information to mark kernels as entry points, correctly setting target triple, data layout, and function calling conventions etc.
- Mapping `__hipsycl_sscp_*` builtins to backend builtins. This typically happens by linking backend-specific bitcode libraries.
- Running optimization passes on the finalized IR
- Lowering the flavored, optimized IR to backend-specific formats, such as ptx or SPIR-V.

For debugging, development, or advanced use cases, each `llvm-to-backend` implementation provides a tool (called `llvm-to-ptx-tool`, `llvm-to-spirv-tool`, ....) that can be invoked to perform the stage 2 compilation step manually.

## Compiler support to accelerate nd_range parallel_for on CPUs (omp.accelerated)

The `nd_range parallel_for` paradigm is not efficiently implementable without explicit compiler support.
AdaptiveCpp however provides multiple options to circumvent this.

1. The recommended and library-only solution is writing kernels using a performance portable
paradigm as AdaptiveCpp's [scoped parallelism](scoped-parallelism.md) extension.
2. If that is not an option (e.g. due to preexisting code), AdaptiveCpp provides a compiler extension that allows efficient execution of the nd_range paradigm at the cost of forcing the host compiler to Clang.
3. Without the Clang plugin, a fiber-based implementation of the `nd_range` paradigm will be used.
However, the relative cost of a barrier in this paradigm is significantly higher compared to e.g. GPU backends. This means that kernels relying on barriers may experience substantial performance degradation, especially if the ratio between barriers and other instructions was tuned for GPUs. Additionally, utilizing barriers in this scenario may prevent vectorization across work items.

For the compiler extension variant, the AdaptiveCpp Clang plugin implements a set of passes to perform deep loop fission
on nd_range parallel_for kernels that contain barriers. The continuation-based synchronization
approach is employed to achieve good performance and functional correctness (_Karrenberg, Ralf, and Sebastian Hack. "Improving performance of OpenCL on CPUs." International Conference on Compiler Construction. Springer, Berlin, Heidelberg, 2012. [https://link.springer.com/content/pdf/10.1007/978-3-642-28652-0_1.pdf](https://link.springer.com/content/pdf/10.1007/978-3-642-28652-0_1.pdf)_).
A deep dive into how the implementation works and why this approach was chosen
can be found in Joachim Meyer's [master thesis](https://joameyer.de/hipsycl/Thesis_JoachimMeyer.pdf).

For more details, see the [installation instructions](installing.md) and the documentation [using AdaptiveCpp](using-hipsycl.md).

## File format for embedded device code

AdaptiveCpp relies on the [heterogeneous container format (HCF)](hcf.md) whenever it takes control over the embedding process of device code (e.g. explicit multipass scenarios).
