#!/bin/bash 
set -e 
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cmake_path=/opt/hipSYCL/llvm/cmake/bin/cmake
HIPSYCL_TEST_LOG_DIR=${HIPSYCL_TEST_LOG_DIR:-/tmp/hipsycl-logs}
mkdir -p $HIPSYCL_TEST_LOG_DIR
HIPSYCL_TEST_CUDA_ARCH=${HIPSYCL_TEST_CUDA_ARCH:-sm_61}
HIPSYCL_TEST_ROCM_ARCH=${HIPSYCL_TEST_ROCM_ARCH:-gfx900}

log_file=${log_file:-HIPSYCL_TEST_LOG_DIR/hipSYCL_image_test-$current_time}
slurm_out=${slurm_out:-$log_file}

targets=( "omp" )
[ "$HIPSYCL_WITH_CUDA" = "ON" ] && targets+=( "cuda:$HIPSYCL_TEST_CUDA_ARCH" )
[ "$HIPSYCL_WITH_ROCM" = "ON" ] && targets+=( "hip:$HIPSYCL_TEST_ROCM_ARCH" )


HIPSYCL_REPO_USER=${HIPSYCL_REPO_USER:-illuhad}
HIPSYCL_REPO_BRANCH=${HIPSYCL_REPO_BRANCH:-develop}

echo "Testing hipSYCL singularity images at $HIPSYCL_PKG_CONTAINER_DIR for targets ${targets[*]}" >> $log_file
echo "Cloning form user $HIPSYCL_REPO_USER branch $HIPSYCL_REPO_BRANCH " >> $log_file

HIPSYCL_TEST_EXCLUDE_FROM_RT="hip:gfx900"
DIR=`pwd`
rm -rf /tmp/hipSYCL-test
git clone -b $HIPSYCL_REPO_BRANCH https://github.com/$HIPSYCL_REPO_USER/hipSYCL.git /tmp/hipSYCL-test
mkdir -p /tmp/hipSYCL-test/tests/build hipSYCL-test
rm -rf /tmp/build
mkdir -p /tmp/build

for distro in archlinux-rolling centos-7 ubuntu-18.04; do
  for target in ${targets[@]}; do
    echo "Starting test for $target for $distro" >> $log_file
    singularity exec --cleanenv  $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro   \
        $cmake_path \
        -DCMAKE_PREFIX_PATH=/opt/hipSYCL/boost/boost \
        -DCMAKE_C_COMPILER=/opt/hipSYCL/llvm/llvm/bin/clang \
        -DCMAKE_CXX_COMPILER=/opt/hipSYCL/llvm/llvm/bin/clang++ \
        -DHIPSYCL_TARGETS=$target \
        -S /tmp/hipSYCL-test/tests \
        -B /tmp/build/ && echo "cmake_succesful (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file \
        || echo "cmake_failed (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file || exit 1  
     

    VERBOSE=1  singularity exec \
        -H /tmp/build $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro \
        make  -j 16 && echo "build_successful (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file \
        || echo "build_failed (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file  || exit 1    

    if [ ! "$target" = $HIPSYCL_TEST_EXCLUDE_FROM_RT ] ;then
      CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
        $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro \
        /tmp/build/sycl_tests \
        && echo "tests_succesful (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file \
        || echo "tests_failed (`wc -l $slurm_out | awk -F' ' '{print $1}' `)" >> $log_file || exit 1 
    else
      echo "test_skipped" >> $log_file
    fi
    rm -rf /tmp/build/* 
done
done
