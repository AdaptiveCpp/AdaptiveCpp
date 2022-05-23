#!/bin/bash
HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-11}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-1}
HIPSYCL_PKG_LLVM_REPO_BRANCH=${HIPSYCL_PKG_LLVM_REPO_BRANCH:-llvmorg-${HIPSYCL_PKG_LLVM_VERSION_MAJOR}.${HIPSYCL_PKG_LLVM_VERSION_MINOR}.${HIPSYCL_PKG_LLVM_VERSION_PATCH}}

HIPSYCL_PKG_LLVM_VERSION=${HIPSYCL_PKG_LLVM_VERSION_MAJOR}.${HIPSYCL_PKG_LLVM_VERSION_MINOR}.${HIPSYCL_PKG_LLVM_VERSION_PATCH}

HIPSYCL_PKG_LLVM_REPO_BRANCH=${HIPSYCL_PKG_LLVM_REPO_BRANCH:-release/9.x}
HIPSYCL_INSTALL_PREFIX=${HIPSYCL_INSTALL_PREFIX:-/opt/hipSYCL}
HIPSYCL_LLVM_BUILD_DIR=${HIPSYCL_LLVM_BUILD_DIR:-$HOME/git/llvm-vanilla}


set -e
if [ -d "$HIPSYCL_LLVM_BUILD_DIR" ]; then
       read -p  "The build directory already exists, do you want to use $HIPSYCL_LLVM_BUILD_DIR anyways?[y]" -n 1 -r
       echo 
       if [[ $REPLY =~ ^[Yy]$ ]]; then
              echo "Using the exisiting directory"
       else
              echo "Please specify a different directory, exiting"
              [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
       fi
else

echo "Cloning LLVM $HIPSYCL_PKG_LLVM_REPO_BRANCH"
git clone -b $HIPSYCL_PKG_LLVM_REPO_BRANCH https://github.com/llvm/llvm-project $HIPSYCL_LLVM_BUILD_DIR
fi

case $HIPSYCL_PKG_LLVM_VERSION in
	9.0.1)
		echo "Applying patch on $HIPSYCL_PKG_LLVM_VERSION"
		sed -i 's/CHECK_SIZE_AND_OFFSET(ipc_perm, mode);//g' $HIPSYCL_LLVM_BUILD_DIR/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cc
		;;
esac


CC=${HIPSYCL_BASE_CC:-clang}
CXX=${HIPSYCL_BASE_CXX:-clang++}
BUILD_TYPE=Release
HIPSYCL_LLVM_INSTALL_PREFIX=$INSTALL_PREFIX/llvm
TARGETS_TO_BUILD="AMDGPU;NVPTX;X86"
NUMTHREADS=`nproc`

CMAKE_OPTIONS="-DLLVM_ENABLE_PROJECTS=clang;compiler-rt;lld;openmp \
               -DOPENMP_ENABLE_LIBOMPTARGET=OFF \
               -DCMAKE_C_COMPILER=$CC \
               -DCMAKE_CXX_COMPILER=$CXX \
               -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
               -DCMAKE_INSTALL_PREFIX=$HIPSYCL_LLVM_INSTALL_PREFIX \
               -DLLVM_ENABLE_ASSERTIONS=OFF \
               -DLLVM_TARGETS_TO_BUILD=$TARGETS_TO_BUILD \
               -DCLANG_ANALYZER_ENABLE_Z3_SOLVER=0 \
               -DLLVM_INCLUDE_BENCHMARKS=0 \
               -DCMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
               -DCMAKE_INSTALL_RPATH=$HIPSYCL_LLVM_INSTALL_PREFIX/lib \
               -DLLVM_ENABLE_OCAMLDOC=OFF \
               -DLLVM_ENABLE_BINDINGS=OFF \
               -DLLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN=OFF \
               -DLLVM_ENABLE_DUMP=OFF"

mkdir -p $HIPSYCL_LLVM_BUILD_DIR/build
cd $HIPSYCL_LLVM_BUILD_DIR/build
cmake $CMAKE_OPTIONS $HIPSYCL_LLVM_BUILD_DIR/llvm
make -j $NUMTHREADS
make install
cp -p $HIPSYCL_LLVM_BUILD_DIR/build/bin/llvm-lit   $HIPSYCL_LLVM_INSTALL_PREFIX/bin/llvm-lit
cp -p $HIPSYCL_LLVM_BUILD_DIR/build/bin/FileCheck  $HIPSYCL_LLVM_INSTALL_PREFIX/bin/FileCheck
cp -p $HIPSYCL_LLVM_BUILD_DIR/build/bin/count      $HIPSYCL_LLVM_INSTALL_PREFIX/bin/count
cp -p $HIPSYCL_LLVM_BUILD_DIR/build/bin/not        $HIPSYCL_LLVM_INSTALL_PREFIX/bin/not
cp -p $HIPSYCL_LLVM_BUILD_DIR/build/bin/yaml-bench $HIPSYCL_LLVM_INSTALL_PREFIX/yaml-bench
