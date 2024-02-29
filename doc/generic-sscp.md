# Generic single-pass compiler (SSCP) implementation

*Generic SSCP* (Single-source, single compiler pass) is AdaptiveCpp's main compiler. Here, kernels are compiled to a generic representation in LLVM IR, which is then lowered at runtime to backend-specific formats such as PTX or SPIR-V as needed. Unlike the interoperability-focused SMCP flows, there is only a single compiler invocation, and the code is only parsed once, no matter how many devices or backends are utilized. This flow can potentially provide *the most portability with the lowest compile times*.

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
- Running optimization passes on the finalized IR.
- Lowering the flavored, optimized IR to backend-specific formats, such as ptx or SPIR-V.

For debugging, development, or advanced use cases, each `llvm-to-backend` implementation provides a tool (called `llvm-to-ptx-tool`, `llvm-to-spirv-tool`, ...) that can be invoked to perform the stage 2 compilation step manually.
