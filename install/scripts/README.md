# hipSYCL installation and packaging scripts

We provide
* Scripts to install hipSYCL and required LLVM, ROCm and CUDA stacks
* Singularity definition files which allow to create singularity container images with hipSYCL
* Scripts to create binary packages of the entire stack for several distributions.

Currently, we support
* Ubuntu 18.04
* CentOS 7
* Arch Linux

## Installing by script
Note that the installation scripts may require the installation of some packages, depending on your distributions. We recommend first looking at the singularity definition files `*.def` for your distribution and installing everything that is installed there. Afterwards, run

* `sudo sh install-llvm.sh` - basic LLVM/clang stack required by hipSYCL
* `sudo sh install-cuda.sh` - downloads and installs a compatible CUDA distribution
* `sudo sh install-rocm.sh` - installs a compatible ROCm stack
* `sudo sh install-hipsycl.sh` - installs hipSYCL.

Unless you have a massive machine, you can expect this to run for half an eternity, so patience is a prerequisite for this installation approach. The easier way is certainly to use our provided binary packages.
The installation prefix can be changed using the environment variable `INSTALL_PREFIX` (default is `/opt/hipSYCL`). Note that the `install-hipsycl.sh` script installs hipSYCL with support for both CUDA and ROCm backends, which means you need to have both installed. At the moment, the installation script does not have an option to disable backends, however this can be easily achieved by editing the `cmake` command inside `install-hipsycl.sh`.

If you change the `INSTALL_PREFIX` to a directory that is writable by your user, `sudo` is not required.

## Building a singularity container
We also provide singularity definition files in order to create singularity container images. Build an image is a two step process. First, create the base image which contains LLVM, ROCm and CUDA stacks, e.g.
```
sudo singularity build base-image.sif base-ubuntu-18.04.def
```
for Ubuntu 18.04. Once this image is built, you can build the actual hipSYCL image:
```
sudo singularity build hipsycl-image.sif hipsycl-ubuntu-18.04.def
```
The process is analogous for the other supported distributions.

## Creating packages
In order to create binary packages for your distribution, you will first have to create container images as described above. Then run (e.g., for Ubuntu):
```
cd packaging; singularity exec hipsycl-image.sif sh make-ubuntu-pkg.sh
```
This script will generate three packages:
* `hipSYCL-base` contains the LLVM stack and clang compiler for hipSYCL. This package must always be installed
* `hipSYCL-rocm` contains a ROCm stack. This must be installed if you wish to target ROCm
* `hipSYCL` contains the actual hipSYCL libraries, headers and tools

Creating CUDA packages is also possible, but this functionality is separate since we do not distribute CUDA binary packages for legal reasons. In order to create a CUDA package, just run the `make-ubuntu-cuda.sh` (for Ubuntu, analogously for other distributions) script. This script can be run on its own and does not require the building the entire stack including container image.
