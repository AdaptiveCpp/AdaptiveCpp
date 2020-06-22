export INSTALL_PREFIX=${INSTALL_PREFIX:-/opt/hipSYCL}

HIPSYCL_PKG_AOMP_RELEASE=${HIPSYCL_PKG_AOMP_RELEASE:-0.7-7}
HIPSYCL_PKG_AOMP_TAG=${HIPSYCL_PKG_AOMP_TAG:-rel_${HIPSYCL_PKG_AOMP_RELEASE}}

set -e
BUILD_DIR=$HOME/git/aomp
rm -rf $BUILD_DIR

#git clone -b hipsycl-0.8 https://github.com/illuhad/aomp $BUILD_DIR/aomp
git clone -b $HIPSYCL_PKG_AOMP_TAG https://github.com/ROCm-Developer-Tools/aomp $BUILD_DIR/aomp

cd $BUILD_DIR/aomp/bin
export CC=${HIPSYCL_BASE_CC:-/opt/hipSYCL/llvm/bin/clang}
export CXX=${HIPSYCL_BASE_CXX:-/opt/hipSYCL/llvm/bin/clang++}
export SUDO=${SUDO:-"disable"}
export AOMP=$INSTALL_PREFIX/rocm
export AOMP_REPOS=$BUILD_DIR
export BUILD_TYPE=Release
#export NVPTXGPUS=60,61,62,70
#export AOMP_BUILD_HIPSYCL_ESSENTIAL=1
export AOMP_BUILD_HIP=1
export CUDA=${CUDA:-$INSTALL_PREFIX/cuda}
#export AOMP_BUILD_CUDA=1

./clone_aomp.sh

case $HIPSYCL_PKG_AOMP_RELEASE in
    0.7-7)
    sed -i 's/openmp pgmath flang flang_runtime//g' $BUILD_DIR/aomp/bin/build_aomp.sh
    sed -i 's/exit 1//g' $BUILD_DIR/aomp/bin/build_hcc.sh
    # This aomp patch to support HIP in conjunction with OpenMP breaks HIP clang printf,
    # so we remove it
    sed -i 's/patch -p1 < $thisdir\/hip.patch//g' $BUILD_DIR/aomp/bin/build_hip.sh

    # Remove problematic -Werror compilation arguments
    sed -i 's/ -Werror//g' $BUILD_DIR/aomp-extras/hostcall/lib/CMakeLists.txt
    sed -i 's/ -Werror//g' $BUILD_DIR/rocr-runtime/src/CMakeLists.txt

    # Remove for compatibility with glibc 2.31
    sed -i 's/CHECK_SIZE_AND_OFFSET(ipc_perm, mode);//g' $BUILD_DIR/llvm-project/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cc
    sed -i 's/CHECK_SIZE_AND_OFFSET(ipc_perm, mode);//g' $BUILD_DIR/hcc/llvm-project/compiler-rt/lib/sanitizer_common/sanitizer_platform_limits_posix.cpp
    ./build_aomp.sh     
    ;;
    11.5-0)
    COMPONENTS="select roct rocr project libdevice extras comgr vdi hipvdi"
    echo "Installing aomp version 11.5-0"
    cd $AOMP_REPOS/amd-llvm-project
    set +e
    # in order to enable the cmake integration of LLD
    git cherry-pick 2b2a961309e65fb175fcf8d56507323fa6872425 
    # This commit breaks clang11
    cd $AOMP_REPOS/rocm-compilersupport
    git revert 6a81326fa9c95f2045064554c471f2568d8c6e7b
    set -e
    cd $BUILD_DIR/aomp/bin
    # So that clang will be properly found
    sed -i "s|-DClang_DIR=$AOMP_REPOS|-DClang_DIR=\$AOMP_REPOS/lib/cmake/clang |" build_comgr.sh
    AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 ./build_aomp.sh $COMPONENTS
    ;;
    11.6-1)
    COMPONENTS="select roct rocr project libdevice extras comgr vdi hipvdi"
    AOMP_CHECK_GIT_BRANCH=0 AOMP_APPLY_ROCM_PATCHES=0 ./build_aomp.sh $COMPONENTS
    ;;
esac
