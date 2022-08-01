#!/bin/bash
set -e
set -o xtrace
HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-11}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-0}
HIPSYCL_HIP_VERSION=${HIPSYCL_HIP_VERSION:-4.0.0}

llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
if [ ! -d ./spack ]; then
  git clone https://github.com/spack/spack.git #-b v0.16.1
  # git clone https://github.com/spack/spack.git spack_upstream
  # echo "upstreams:
  # spack-instance-1:
  #   install_tree: /root/spack_upstream/opt/spack" > /root/spack/etc/spack/defaults/upstreams.yaml
fi
export SPACK_ROOT=~/spack
export PATH=$SPACK_ROOT/bin:$PATH
. $SPACK_ROOT/share/spack/setup-env.sh

sed -i 's|root: .*$|root: /opt/hipSYCL/rocm/|' spack/etc/spack/defaults/config.yaml
sed -i 's|all: .*$|all: ${PACKAGE}|' spack/etc/spack/defaults/config.yaml
sed -i 's|# build_jobs: .*$|build_jobs: 16|' spack/etc/spack/defaults/config.yaml
. $SPACK_ROOT/share/spack/setup-env.sh
spack compiler find /opt/hipSYCL/llvm/llvm/bin/

# Somteimes some parallel instances exit due to waiting too long for a lock
# In case that happens we run the sequential version to check if everything have been
# installed properly
parallel --lb -N0 spack install hip@$HIPSYCL_HIP_VERSION%clang@$llvm_version target=x86_64 ::: {1..16} || error="1"
if [ "$error" = "1" ]; then
  spack install hip%clang@$llvm_version target=x86_64
fi
spack gc -y
