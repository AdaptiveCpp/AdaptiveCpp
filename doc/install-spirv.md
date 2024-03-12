# AdaptiveCpp installation instructions for SPIR-V devices with Level Zero

**Note: Targeting SPIR-V devices through OpenCL is currently more mature and may yield better results.**

Please install the Level Zero loader and a Level Zero driver such as the Intel [compute runtime](https://github.com/intel/compute-runtime) for Intel GPUs.

The Level Zero backend can be enabled using `cmake -DWITH_LEVEL_ZERO_BACKEND=ON` when building AdaptiveCpp.



