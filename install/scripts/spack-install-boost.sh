#!/bin/bash
set -e
HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-11}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-0}

llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
if [ ! -d ./spack ]; then
  git clone https://github.com/spack/spack.git
fi
export SPACK_ROOT=/root/spack
export PATH=$SPACK_ROOT/bin:$PATH
. $SPACK_ROOT/share/spack/setup-env.sh

sed -i 's|root: .*$|root: /opt/hipSYCL/boost/|' spack/etc/spack/defaults/config.yaml
sed -i 's|all: .*$|all: ${PACKAGE}|' spack/etc/spack/defaults/config.yaml
sed -i 's|# build_jobs: .*$|build_jobs: 64|' spack/etc/spack/defaults/config.yaml
. $SPACK_ROOT/share/spack/setup-env.sh
spack compiler find /opt/hipSYCL/llvm/llvm/bin/
parallel --lb -N0 spack install boost%clang@$llvm_version context=True fiber=True target=x86_64 cxxstd=11 ::: {1..16}
spack gc -y

