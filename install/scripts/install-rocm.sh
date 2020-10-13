export HIPSYCL_INSTALL_PREFIX=${HIPSYCL_INSTALL_PREFIX:-/opt/hipSYCL}

HIPSYCL_PKG_AOMP_RELEASE=${HIPSYCL_PKG_AOMP_VERSION:-0.7-7}
HIPSYCL_PKG_AOMP_TAG=${HIPSYCL_PKG_AOMP_TAG:-rel_${HIPSYCL_PKG_AOMP_RELEASE}}

set -e
HIPSYCL_ROCM_BUILD_DIR=${HIPSYCL_ROCM_BUILD_DIR:-$HOME/git/aomp}

export CC=${HIPSYCL_BASE_CC:-clang}
export CXX=${HIPSYCL_BASE_CXX:-clang++}
export SUDO=${SUDO:-"disable"}
export AOMP=$HIPSYCL_INSTALL_PREFIX/rocm
export BUILD_TYPE=Release
#export NVPTXGPUS=60,61,62,70
#export AOMP_BUILD_HIPSYCL_ESSENTIAL=1
export AOMP_BUILD_HIP=1
export CUDA=${CUDA:-$HIPSYCL_INSTALL_PREFIX/cuda}
#export AOMP_BUILD_CUDA=1

if [ -d "$HIPSYCL_ROCM_BUILD_DIR" ]; then
       read -p  "The build directory already exists, do you want to use $HIPSYCL_ROCM_BUILD_DIR anyways?[y]" -n 1 -r
       echo 
       if [[ $REPLY =~ ^[Yy]$ ]]; then
              echo "Using the exisiting directory"
       else
              echo "Please specify a different directory, exiting"
              [[ "$0" = "$BASH_SOURCE" ]] && exit 1 || return 1
       fi
else
echo "Cloning aomp"
git clone -b $HIPSYCL_PKG_AOMP_TAG https://github.com/ROCm-Developer-Tools/aomp $HIPSYCL_ROCM_BUILD_DIR/aomp
cd $HIPSYCL_ROCM_BUILD_DIR/aomp/bin
./clone_aomp.sh
fi


cd $HIPSYCL_ROCM_BUILD_DIR/aomp/bin
case $HIPSYCL_PKG_AOMP_RELEASE in
	0.7-7)
    sed -i 's/openmp pgmath flang flang_runtime//g' $HIPSYCL_ROCM_BUILD_DIR/aomp/bin/build_aomp.sh
    sed -i 's/exit 1//g' $HIPSYCL_ROCM_BUILD_DIR/aomp/bin/build_hcc.sh
    # This aomp patch to support HIP in conjunction with OpenMP breaks HIP clang printf,
    # so we remove it
    sed -i 's/patch -p1 < $thisdir\/hip.patch//g' $HIPSYCL_ROCM_BUILD_DIR/aomp/bin/build_hip.sh

    # Remove problematic -Werror compilation arguments
    sed -i 's/ -Werror//g' $HIPSYCL_ROCM_BUILD_DIR/aomp-extras/hostcall/lib/CMakeLists.txt
    sed -i 's/ -Werror//g' $HIPSYCL_ROCM_BUILD_DIR/rocr-runtime/src/CMakeLists.txt

    # Remove for compatibility with glibc 2.31
    sed -i 's/CHECK_SIZE_AND_OFFSET(ipc_perm, mode);//g' $HIPSYCL_ROCM_BUILD_DIR/llvm-project/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cc
    sed -i 's/CHECK_SIZE_AND_OFFSET(ipc_perm, mode);//g' $HIPSYCL_ROCM_BUILD_DIR/hcc/llvm-project/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cpp
  ;;
esac
./build_aomp.sh
