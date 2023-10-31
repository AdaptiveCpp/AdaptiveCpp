# AdaptiveCpp installation instructions for SPIR-V/Level Zero

Please install the Level Zero loader and a Level Zero driver such as the Intel [compute runtime](https://github.com/intel/compute-runtime) for Intel GPUs.

The Level Zero backend can be enabled using `cmake -DWITH_LEVEL_ZERO_BACKEND=ON` when building AdaptiveCpp.


## For the legacy SPIR-V compilation flow (`--acpp-targets=spirv`)
**Please ignore the following if you actually want to use the more modern generic single-pass compiler (`--acpp-targets=generic`), which is recommended.**

Please build AdaptiveCpp against a clang/LLVM that has Intel's patches to generate SPIR-V, following the [LLVM installation instructions](install-llvm.md). Once all required patches are upstreamed this will work with regular clang distributions; until then AdaptiveCpp needs to be built against DPC++/Intel's LLVM [fork](https://github.com/intel/llvm).
Unfortunately, the binary distribution of DPC++ do not contain development headers, so the clang plugin required by the CUDA and ROCm backends cannot be compiled, but the open source fork should be able to also target CUDA and ROCm.

