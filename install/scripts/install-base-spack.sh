#!/bin/bash
set -e

llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
git clone https://github.com/spack/spack.git
export SPACK_ROOT=/spack
export PATH=$SPACK_ROOT/bin:$PATH

sed -i 's|root: .*$|root: /opt/hipSYCL/llvm/|' spack/etc/spack/defaults/config.yaml
sed -i 's|all: .*$|all: ${PACKAGE}|' spack/etc/spack/defaults/config.yaml
sed -i 's|# build_jobs: .*$|build_jobs: 64|' spack/etc/spack/defaults/config.yaml
spack compiler find /opt/hipSYCL/llvm/llvm/bin/
spack install llvm@$llvm_version libcxx=False 

sed -i 's|root: .*$|root: /opt/hipSYCL/boost/|' spack/etc/spack/defaults/config.yaml
spack compiler find /opt/hipSYCL/llvm/llvm/bin/
spack install boost%clang@$llvm_version

sed -i 's|root: .*$|root: /opt/hipSYCL/rocm/|' spack/etc/spack/defaults/config.yaml
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
spack compiler find /opt/hipSYCL/llvm/llvm/bin/
spack install hip%clang@$llvm_version



