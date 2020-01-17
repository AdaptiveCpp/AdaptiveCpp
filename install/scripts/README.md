# hipSYCL installation and packaging scripts

We provide
* Scripts to install hipSYCL and required LLVM, ROCm and CUDA stacks
* Repositories for all supported distributions
* Singularity definition files which allow to create singularity container images with hipSYCL
* Scripts to create binary packages of the entire stack for several distributions.

Currently, we support
* Ubuntu 18.04
* CentOS 7
* Arch Linux

## Installing from repositories
Installing using the repositories is beneficial because the hipSYCL installation can be kept up to date with regular system updates.

### Ubuntu 18.04
Add the hipSYCL repo to the sources:  
`echo "deb http://repo.urz.uni-heidelberg.de/sycl/deb/ ./bionic main" > /etc/apt/sources.list.d/hipsycl.list`  
Import the pgp public key:  
`wget -q -O - http://repo.urz.uni-heidelberg.de/sycl/hipsycl.asc | apt-key add -`  
After updating the packages can be installed with apt  
`apt update`  
`apt install <hipSYCL package>`  

### Centos
The following command will set up everything:  
`yum-config-manager --add-repo http://repo.urz.uni-heidelberg.de/sycl/rpm/centos7/hipsycl.repo`  
  
After an update the packages can be installed with yum  
`yum update`  
`yum install <hipSYCL package>`  

Note: hipSYCL depends on devtoolset-7 which is available in the scl repository:  
`yum install centos-release-scl`  
`yum update`  

### Archlinux
The following should be added to `/etc/pacman.conf`:
```
[hipsycl]   
Server = http://repo.urz.uni-heidelberg.de/sycl/archlinux/x86_64
```
In case that pacman couldn't fetch the public key from a key server, you can download it from:
http://repo.urz.uni-heidelberg.de/sycl/  
And then add it manually.   
(see: [Arch wiki](https://wiki.archlinux.org/index.php/Pacman/Package_signing#Adding_unofficial_keys))
  
After upgrade the packages can be installed with pacman  
`pacman -Sy`  
Pacman should fetch the public key from a key server, however it needs to be signed locally first:
`pacman-key --lsign E967BA09716F870320089583E68CC4B9B2B75080`  
Now update should finish without an error  
`pacman -Sy`  
`pacman -S <hipSYCL package>`

## Installing by script
Note that the installation scripts may require the installation of some packages, depending on your distributions. We recommend first looking at the singularity definition files `*.def` for your distribution and installing everything that is installed there. Afterwards, run

* `sudo sh install-llvm.sh` - basic LLVM/clang stack required by hipSYCL
* `sudo sh install-cuda.sh` - downloads and installs a compatible CUDA distribution
* `sudo sh install-rocm.sh` - installs a compatible ROCm stack
* `sudo sh install-hipsycl.sh` - installs hipSYCL.

Unless you have a massive machine, you can expect this to run for half an eternity, so patience is a prerequisite for this installation approach. The easier way is certainly to use our provided binary packages.
The installation prefix can be changed using the environment variable `INSTALL_PREFIX` (default is `/opt/hipSYCL`). Note that the `install-hipsycl.sh` script builds hipSYCL with support for both CUDA and ROCm backends by default, which means you need to have both installed. If you wish to disable support for CUDA/ROCm, set the `HIPSYCL_WITH_CUDA` or `HIPSYCL_WITH_ROCM` environment variables to `OFF`.

If you change the `INSTALL_PREFIX` to a directory that is writable by your user, `sudo` is not required.

## Building a singularity container
We also provide singularity definition files in order to create singularity container images. Building an image is a two step process. First, create the base image which contains LLVM, ROCm and CUDA stacks, e.g.
```
sudo singularity build base-ubuntu-18.04.sif base-ubuntu-18.04.def
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
Note: If you only intend to install hipSYCL's CUDA stack on a single machine for home use, it may be easier and faster to just install it directly using the install script: Run
```
sudo sh install-cuda.sh 
```
which will install it directly to `/opt/hipSYCL/cuda` where hipSYCL expects it.
