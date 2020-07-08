# hipSYCL installation notes for CUDA
Installing hipSYCL for CUDA is relatively straight forward. The following requirements must be met:
* clang must be built without the `_DEBUG` macro defined. Compiling clang with `_DEBUG` is a configuration that hipSYCL does not support due to incorrect `assert()` statements inside clang that get triggered due to the - from clang's perspective - slightly unusual and unexpected way in which hipSYCL employs the HIP/CUDA toolchains. Regular clang release builds as they can be found in distributions usually meet this requirement.
* clang must have been compiled with the NVPTX backend enabled (which is usually the case)
* Please check in the documentation of your clang version if your CUDA version is supported by your clang installation.

The following cmake variables may be relevant:
* Use `CUDA_TOOLKIT_ROOT_DIR` to point hipSYCL to the CUDA root installation directory (e.g. `/usr/local/cuda`), if cmake doesn't find the right CUDA installation.
