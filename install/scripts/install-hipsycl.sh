#!/bin/bash
export HIPSYCL_INSTALL_PREFIX=${HIPSYCL_INSTALL_PREFIX:-/opt/hipSYCL}

set -e
HIPSYCL_BUILD_DIR=${HIPSYCL_BUILD_DIR:-/tmp/hipSYCL-installer}
HIPSYCL_REPO_USER=${HIPSYCL_REPO_USER:-illuhad}
HIPSYCL_REPO_BRANCH=${HIPSYCL_REPO_BRANCH:-develop}
HIPSYCL_WITH_CUDA=${HIPSYCL_WITH_CUDA:-ON}
HIPSYCL_WITH_ROCM=${HIPSYCL_WITH_ROCM:-ON}
HIPSYCL_LLVM_DIR=${HIPSYCL_LLVM_DIR:-/opt/hipSYCL/llvm/lib/}

if [ -d "$HIPSYCL_BUILD_DIR" ]; then
       read -p  "The build directory already exists, do you want to use $HIPSYCL_BUILD_DIR anyways?[y]" -n 1 -r
       echo 
       if [[ ! $REPLY =~ ^[Yy]$ ]]; then
              echo "Please specify a different directory, exiting"
              [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
       else
              echo "Using the exisiting directory"
       fi
else
echo "Cloning hipSYCL"
git clone --recurse-submodules -b $HIPSYCL_REPO_BRANCH https://github.com/$HIPSYCL_REPO_USER/hipSYCL $HIPSYCL_BUILD_DIR

fi

mkdir -p $HIPSYCL_BUILD_DIR/build
cd $HIPSYCL_BUILD_DIR/build

cmake \
-DWITH_CPU_BACKEND=ON \
-DWITH_CUDA_BACKEND=$HIPSYCL_WITH_CUDA \
-DWITH_ROCM_BACKEND=$HIPSYCL_WITH_ROCM \
-DLLVM_DIR=$HIPSYCL_LLVM_DIR \
-DROCM_PATH=$HIPSYCL_INSTALL_PREFIX/rocm \
-DCMAKE_INSTALL_PREFIX=$HIPSYCL_INSTALL_PREFIX \
..

make -j `nproc` install
