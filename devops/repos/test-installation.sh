#!/bin/bash 
set -e 
set -o xtrace
set -xv
if [ "$#" -ne 3 ]; then
  echo "
  This script is responsible for testing the installation inside a built container containing a hipSYCL installation
  the following tests are built and executed: sycl_tests
  
  
  usage:
   <dir_of_test_script> <distro> <backend>
  dir_of_test_scripts: Points to the directory where this script is located
  distro: The distribution for which the packages are suposed to be tested
  backend: A bitmask of the enabled backends, from leat to most important bit: CUDA,ROCM. 1 means enabled 0 means disabled
  
  Important ENV variables:
    - HIPSYCL_TEST_DIR: The location where the test containers will be installed
    - HIPSYCL_TEST_EXCLUDE_FROM_RT: by default set to hip:gfx900. For this backend, we only build the tests.
  "
  exit -1
fi
cd $1
distro=$2
backend=$3
HIPSYCL_WITH_CUDA="OFF" 
HIPSYCL_WITH_ROCM="OFF"
if [[ ${backend:0:1} = "1" ]]; then HIPSYCL_WITH_ROCM="ON"; else HIPSYCL_WITH_ROCM="OFF"; fi
if [[ ${backend:1:2} = "1" ]]; then HIPSYCL_WITH_CUDA="ON"; else HIPSYCL_WITH_CUDA="OFF"; fi
source ./common/init.sh
current_time=$(date "+%Y.%m.%d-%H.%M.%S")

cmake_path=/opt/hipSYCL/llvm/cmake/bin/cmake
HIPSYCL_TEST_LOG_DIR=${HIPSYCL_TEST_LOG_DIR:-/tmp/hipsycl-logs}
mkdir -p $HIPSYCL_TEST_LOG_DIR
HIPSYCL_TEST_CUDA_ARCH=${HIPSYCL_TEST_CUDA_ARCH:-sm_61}
HIPSYCL_TEST_ROCM_ARCH=${HIPSYCL_TEST_ROCM_ARCH:-gfx900}

log_file=${log_file:-$HIPSYCL_TEST_LOG_DIR/hipSYCL_image_test-$current_time}
touch $log_file
slurm_out=${slurm_out:-$log_file}

targets=( "omp" )
[ "$HIPSYCL_WITH_CUDA" = "ON" ] && targets+=( "cuda:$HIPSYCL_TEST_CUDA_ARCH" )
[ "$HIPSYCL_WITH_ROCM" = "ON" ] && targets+=( "hip:$HIPSYCL_TEST_ROCM_ARCH" )


HIPSYCL_REPO_USER=${HIPSYCL_REPO_USER:-illuhad}
HIPSYCL_REPO_BRANCH=${HIPSYCL_REPO_BRANCH:-develop}

echo "Testing hipSYCL singularity images at $HIPSYCL_PKG_CONTAINER_DIR for targets ${targets[*]}" >> $log_file
echo "Cloning form user $HIPSYCL_REPO_USER branch $HIPSYCL_REPO_BRANCH " >> $log_file

HIPSYCL_TEST_EXCLUDE_FROM_RT=${HIPSYCL_TEST_EXCLUDE_FROM_RT:-"hip:gfx900"}
DIR=`pwd`

mkdir -p /tmp/hipSYCL-test/tests/build 
mkdir -p /tmp/build/$distro-$backend

for target in ${targets[@]}; do
  echo "Starting test for $target for $distro" >> $log_file
  singularity exec --cleanenv  $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro-$backend   \
      $cmake_path \
      -DCMAKE_PREFIX_PATH=/opt/hipSYCL/boost/boost \
      -DCMAKE_C_COMPILER=/opt/hipSYCL/llvm/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/hipSYCL/llvm/llvm/bin/clang++ \
      -DHIPSYCL_TARGETS=$target \
      -S /tmp/hipSYCL-test/tests \
      -B /tmp/build/$distro-$backend
   

  VERBOSE=1 CUDA_VISIBLE_DEVICES=0 singularity exec --nv \
      -H /tmp/build/$distro-$backend $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro-$backend \
      make  -j 16

  if [ ! "$target" = $HIPSYCL_TEST_EXCLUDE_FROM_RT ] ;then
    #CUDA_VISIBLE_DEVICES=0 \
	    singularity exec --nv \
      $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-$distro-$backend \
      /tmp/build/$2-$3/sycl_tests 
  else
    echo "test_skipped" >> $log_file
  fi
  rm -rf /tmp/build/$2-$3
done
