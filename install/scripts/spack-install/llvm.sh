#!/bin/bash
set -e
set -o xtrace
HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-11}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-0}

llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
if [ ! -d ./spack ]; then
  git clone https://github.com/spack/spack.git #-b v0.16.1
  # git clone https://github.com/spack/spack.git spack_upstream
  # sed -i 's|- $spack/var/spack/repos/builtin|- /root/spack_upstream/var/spack/repos/builtin|' spack/etc/spack/defaults/repos.yaml
fi
export SPACK_ROOT=/root/spack
export PATH=$SPACK_ROOT/bin:$PATH
. $SPACK_ROOT/share/spack/setup-env.sh
spack compiler find || echo "No new compilers"; spack compilers

sed -i 's|root: .*$|root: /opt/hipSYCL/llvm/|' spack/etc/spack/defaults/config.yaml
sed -i 's|all: .*$|all: ${PACKAGE}|' spack/etc/spack/defaults/config.yaml
sed -i 's|# build_jobs: .*$|build_jobs: 16|' spack/etc/spack/defaults/config.yaml
sed -i 's|projects.append("clang-tools-extra")|#projects.append("clang-tools-extra")|' spack/var/spack/repos/builtin/packages/llvm/package.py

. $SPACK_ROOT/share/spack/setup-env.sh
llvm_version=$HIPSYCL_PKG_LLVM_VERSION_MAJOR.$HIPSYCL_PKG_LLVM_VERSION_MINOR.$HIPSYCL_PKG_LLVM_VERSION_PATCH
parallel --lb -N0 spack install llvm@$llvm_version cuda=False libcxx=True polly=False lldb=False lld=True internal_unwind=False gold=False target=x86_64 build_type=MinSizeRel -flang ::: {1..16} || error=1
if [ "$error" = "1" ]; then
  spack install llvm@$llvm_version -flang cuda=False libcxx=True polly=False lldb=False lld=True internal_unwind=False gold=False target=x86_64 build_type=MinSizeRel
fi
#spack install llvm@$llvm_version cuda=False libcxx=False target=x86_64
spack load llvm
spack compiler find /opt/hipSYCL/llvm/llvm/
spack unload llvm
