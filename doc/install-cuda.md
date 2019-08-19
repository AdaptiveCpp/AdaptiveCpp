# hipSYCL installation notes for CUDA
Installing hipSYCL for CUDA is relatively straight forward. The following requirements must be met:
* clang must be built without the `_DEBUG` macro defined. Compiling clang with `_DEBUG` is a configuration that hipSYCL does not support due to incorrect `assert()` statements inside clang that get triggered due to the - from clang's perspective - slightly unusual and unexpected way in which hipSYCL employs the HIP/CUDA toolchains. Regular clang release builds as they can be found in distributions usually meet this requirement.
* clang must have been compiled with the NVPTX backend enabled (which is usually the case)
* Note that clang 8 only supports CUDA 9.x.
* Note that clang 9 supports CUDA 10. However, CUDA 10.1 is known to cause runtime crashes for CUDA code compiled with clang 9 in some cases. We therefore recommend CUDA 10.0.

The following cmake variables may be relevant:
* Use `CUDA_TOOLKIT_ROOT_DIR` to point hipSYCL to the CUDA root installation directory (e.g. `/usr/local/cuda`), if cmake doesn't find the right CUDA installation.
