#!/bin/bash 
set -e 
set -o xtrace
source ~/envs.out
if [ "$#" -lt 4 ]; then
  echo "
  This script is responsible for creating a base image (base), adding the hipSYCL repo (add_repo),
  installing the hipSYCL package and its dependencies (install_dependencies), and then running the tests
  (run_tests) and eventually cleaning up (clean_up) for the specified distribution and backend combination

  usage:
   <dir_of_test_script> <distro> <backends> [action: build, add_repo, intall_dependencies, run_test, clean_up] <target_repo>
  
  dir_of_test_scripts: Points to the directory where this script is located
  distro: The distribution for which the packages are supposed to be tested
  backends: A bitmask of the enabled backends, from leat to most important bit: CUDA,ROCM. 1 means enabled 0 means disabled
  actions: build: build a container image for the specified distribution:
                      it creates an image in the directory: HIPSYCL_TEST_DIR/distro-backend folder
           add_repo: run the ../../install/scripts/add-repo-<distro>.sh script to add the hipSYCL repo to the base image
           install_dependencis: Installs the version of hipSYCL with the targeted backends, in case of Cuda backend, is tested, install Cuda from an external source see
                      ../../install/scripts/spack-install-Cuda.sh
           run_test: executes the ./test-installation.sh script in the built singularity container.
           clean_up: Useful if the container is going to be reused. Deletes all installed packages and Cuda if necessary.
  target_repo: an optional path to the repository directory from the hipSYCL base repo. useful if testing experimental repos

  Important ENV variables:
    - HIPSYCL_TEST_DIR: The location where the test containers will be installed
  "
  exit -1
fi
home_dir=$1
distro=$2
backends=$3
action=$4
target_repo=$5

HIPSYCL_WITH_CUDA="OFF" 
HIPSYCL_WITH_ROCM="OFF"
if [[ ${backends:0:1} = "1" ]]; then HIPSYCL_WITH_ROCM="ON"; else HIPSYCL_WITH_ROCM="OFF"; fi
if [[ ${backends:1:2} = "1" ]]; then HIPSYCL_WITH_CUDA="ON"; else HIPSYCL_WITH_CUDA="OFF"; fi

cd $home_dir
source ./common/init.sh
#slurm_out=$1/slurm-$SLURM_JOB_ID.out
#target_repo=${2:-""}

echo $slurm_out 
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
HIPSYCL_TEST_DIR=${HIPSYCL_TEST_DIR:-/tmp/hipsycl-test/}
echo $HIPSYCL_TEST_DIR
HIPSYCL_PKG_TYPE=${HIPSYCL_PKG_TYPE:-"-nightly"}
mkdir -p $HIPSYCL_TEST_DIR
export slurm_out

dict_key="$HIPSYCL_WITH_ROCM$HIPSYCL_WITH_CUDA"
echo "Starting comprehensive testing of the package repositories for ${distros[*]}"

if [ "$action" = "build" ];then
  singularity build --fakeroot --force --sandbox  $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends ./definitions-test-containers/$distro.def


elif [ "$action" = "add_repo" ]; then
  singularity exec --fakeroot --writable  -B ../../install/scripts:/mnt \
    $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends sh /mnt/add-hipsycl-repo/$distro.sh $target_repo 


elif [ "$action" = "install_dep" ]; then
  if [ "$HIPSYCL_WITH_CUDA" = "ON" ]; then
       singularity exec --fakeroot --writable  -B ../../install/scripts:/mnt \
         $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends sh /mnt/spack-install/cuda.sh 
  fi
  singularity exec --fakeroot --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends \
      ${install_cmd[$distro]}${pkg_suffix[$dict_key]}-$HIPSYCL_PKG_TYPE 


elif [ "$action" = "run_tests" ]; then
  export HIPSYCL_WITH_CUDA
  export HIPSYCL_WITH_ROCM 
  echo "Start testing" 
  HIPSYCL_PKG_CONTAINER_DIR=$HIPSYCL_TEST_DIR 
  export HIPSYCL_PKG_CONTAINER_DIR
  `pwd`/test-installation.sh `pwd` $distro $backends 


elif [ "$action" = "clean_up" ]; then
  singularity exec --fakeroot --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends \
      ${cleanup_cmd[$distro]}${pkg_suffix[$dict_key]}-$HIPSYCL_PKG_TYPE
  
  singularity exec --fakeroot --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends \
      ${cleanup_dep[$distro]} 

  singularity exec --fakeroot --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro-$backends rm -rf /opt/hipSYCL/cuda
fi
