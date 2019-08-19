# Automated, reproducible build and test infrastructure for hipSYCL

`build.py` implements an infrastructure to automatically build and test hipSYCL for all supported platforms in a reproducible manner. It relies on the container solution [singularity](https://sylabs.io/docs/). It is recommended that, before submitting a pull request, the code is first tested with `build.py` since this can allow the detection of problems at a very early phase.

## Prerequisites
* `singularity` >= 3.0
* sudo privileges. Building a container with singularity requires root privileges because singularity needs to set up container isolation.
* python 3.
* Internet access.

## How to use

There are two main ways of using `build.py`:
* `python3 ./build.py` builds and tests a singularity image containing hipSYCL built from the local source tree in which `build.py` resides. This is useful to check own developments before publishing them.
* `python3 ./build.py <user> <branch>` builds and tests the specified branch of the source code at `https://github.com/user/hipSYCL`. This is useful e.g. when reviewing pull requests.

Building the image is done in two steps. First, a base image containing CUDA, ROCm and LLVM compiler stacks is created. This image, once created, will be reused for all future runs which can greatly speed up testing (rebuilding can be forced by deleting it again).
Next, the actual hipSYCL image is created on top of the base image. hipSYCL is compiled (including unit tests) for all three supported platforms CPU, CUDA and ROCm.
Once hipSYCL has been compiled successfully, tests are carried out. While compilation tests are always carried out, tests that require actually executing hipSYCL program can be disabled using the `HIPSYCL_NO_RUNTIME_TESTING` environment variable. This can be useful if you do not have both CUDA and ROCm GPUs available, preventing a successful run of the test cases for all platforms. For example,
```
HIPSYCL_NO_RUNTIME_TESTING=cpu,rocm python3 build.py 
```
will only execute tests that have been compiled for CUDA.

By default, the tests will be compiled for `sm_52` (CUDA) and `gfx906` (ROCm). If this does not fit your hardware, you can change this with the `CUDA_GPU_ARCH` and `ROCM_GPU_ARCH` environment variables.

The built singularity images will be stored in `../../hipsycl-singularity-build`.
