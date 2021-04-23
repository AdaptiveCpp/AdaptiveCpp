#!/bin/bash
set -e
set -o xtrace
HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-11}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-0}

llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
if [ ! -d ./spack ]; then
  git clone https://github.com/spack/spack.git #-b v0.16.1
fi
export SPACK_ROOT=/root/spack
export PATH=$SPACK_ROOT/bin:$PATH
. $SPACK_ROOT/share/spack/setup-env.sh

sed -i 's|root: .*$|root: /opt/hipSYCL/boost/|' spack/etc/spack/defaults/config.yaml
sed -i 's|all: .*$|all: ${PACKAGE}|' spack/etc/spack/defaults/config.yaml
sed -i 's|# build_jobs: .*$|build_jobs: 16|' spack/etc/spack/defaults/config.yaml
. $SPACK_ROOT/share/spack/setup-env.sh
spack compiler find /opt/hipSYCL/llvm/llvm/bin/
# Spack distributed build in this form causes Timeouts sometimes.... maybe use a upstream solution... yeah probably.... 

parallel --joblog /tmp/spack-install-boost.exit --lb -N0 spack install boost%clang@$llvm_version context=True fiber=True target=x86_64 cxxstd=11 ::: {1..16} || error=1
if [ "$error" = "1" ]; then 
  spack install boost%clang@$llvm_version context=True fiber=True target=x86_64 cxxstd=11
fi
spack gc -y

