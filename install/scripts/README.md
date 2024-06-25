# hipSYCL installation and packaging scripts

We provide
* Scripts to install hipSYCL and required LLVM, ROCm and CUDA stacks
* Repositories for all supported distributions
* Singularity definition files which allow to create singularity container images with hipSYCL
* Pre-built singularity containers
* Scripts to create binary packages of the entire stack for several distributions.

Currently, we support
* Ubuntu 18.04
* Ubuntu 20.04
* CentOS 7
* Arch Linux

## Installing from repositories
Installing using the repositories is beneficial because the hipSYCL installation can be kept up to date with regular system updates. We provide stable packages subject to more rigorous testing and nightly packages built from the current development head.

We provide the following packages in both versions:

Base packages: 
* `hipSYCL-base<-nightly>`
* `hipSYCL-base-rocm<-nightly>`
HipSYCL packages:
* `hipSYCL-omp<-nightly>`
* `hipSYCL-omp-cuda<-nightly>`
* `hipSYCL-omp-rocm<-nightly>`
* `hipSYCL-omp-rocm-cuda<-nightly>`
Two meta-packages in order to keep consistent with the previous packages:
* `hipSYCL-full<-nightly>` -> `hipSYCL-omp-rocm-cuda<-nightly>`
* `hipSYCL<-nightly>` -> `hipSYCL-omp-rocm-cuda<-nightly>`

We require some additional software repos to be enabled (for example, `release-scl` and `epel` for centos 7 ). To make adding these easier, we provide scripts in the `install/scripts/add-hipsycl-repo` for all supported distributions that handles adding these repositories, as well as adding the hipSYCL repo.

## Installing by script
Note that the installation scripts may require the installation of some packages, depending on your distributions. We recommend first looking at the singularity definition files `*.def` for your distribution and installing everything that is installed there. Afterwards, run

* `sudo sh install-llvm.sh` - basic LLVM/clang stack required by hipSYCL
* `sudo sh install-cuda.sh` - downloads and installs a compatible CUDA distribution
* `sudo sh install-rocm.sh` - installs a compatible ROCm stack
* `sudo sh install-hipsycl.sh` - installs hipSYCL.

Unless you have a massive machine, you can expect this to run for half an eternity, so patience is a prerequisite for this installation approach. The easier way is to use our provided binary packages.
The installation prefix can be changed using the environment variable `INSTALL_PREFIX` (default is `/opt/hipSYCL`). Note that the `install-hipsycl.sh` script builds hipSYCL with support for both CUDA and ROCm backends by default, which means you need to have both installed. If you wish to disable support for CUDA/ROCm, set the `HIPSYCL_WITH_CUDA` or `HIPSYCL_WITH_ROCM` environment variables to `OFF`.

If you change the `INSTALL_PREFIX` to a directory that is writable by your user, `sudo` is not required.

## Building a singularity container
We also provide singularity definition files in order to create singularity container images. Building an image consists of building a writable base image and afterwards installing the dependencies and hipsycl into the container

```
singularity build --fakeroot --sandbox base-ubuntu-18.04.sif base-definitions/base-ubuntu-18.04.def
```
for Ubuntu 18.04. Once this image is built, you can start adding the dependencies
```
singularity exec hipsycl-ubuntu-18.04.def install-llvm.sh
singularity exec hipsycl-ubuntu-18.04.def install-rocm.sh
singularity exec hipsycl-ubuntu-18.04.def install-cuda.sh
```
Note that there are two type of installation scripts available at the moment the regular ones located in the `install/scripts/` directory, and scripts that use spack to install the dependencies located in `install/scripts/spack-install/`. The spack install scripts are well tested, therefore we recommend using those for the installation. The regular install scripts might need some changes to work flawlessly.

## Pre-built singularity containers

We provide pre-built singularity images for all supported distributions. The containers are available here: http://repo.urz.uni-heidelberg.de/sycl/singularity/ 

The images are validated by building the hipSYCL unit tests for all supported backends, and running them for OpenMP and CUDA.

Please note that due to legal reasons, the images do not contain the CUDA installation. Please use the `install/scripts/install-cuda.sh` script to install it afterwards. Note that this is only possible in case the container is writable; therefore we recommend installing CUDA by executing the following commands:

```
singularity shell build --sandbox --fakeroot <container_name>.sif <container_name>
singularity exec --writable --fakeroot <container_name> bash install/scripts/install-cuda.sh
```

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
Note: If you only intend to install hipSYCL's CUDA stack on a single machine for home use, it may be easier and faster to just install it directly using the install script: Run
```
sudo sh install-cuda.sh 
```
which will install it directly to `/opt/hipSYCL/cuda` where hipSYCL expects it.
