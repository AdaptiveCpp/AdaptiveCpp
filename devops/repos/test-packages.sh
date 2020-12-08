#!/bin/bash 
set -e 
cd $1
source ./common/init.sh
slurm_out=$1/slurm-$SLURM_JOB_ID.out
echo $slurm_out 
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
HIPSYCL_TEST_LOG_DIR=${HIPSYCL_TEST_LOG_DIR:-/tmp/hipsycl-logs}
log_file=$HIPSYCL_TEST_LOG_DIR/hipSYCL_pkg_test-$current_time
HIPSYCL_TEST_DIR=${HIPSYCL_TEST_DIR:-/tmp/hipsycl-test/}
HIPSYCL_PKG_TYPE=${HIPSYCL_PKG_TYPE:-"-nightly"}
mkdir -p $HIPSYCL_TEST_DIR
export log_file 
export slurm_out

echo "Starting comprehensive testing of the package repositories for ${distros[*]}" >> $log_file

for distro in ${distros[@]}; do
  echo "Building new image for $distro" >> $log_file
  sudo singularity build --force --sandbox  $HIPSYCL_TEST_DIR/hipsycl-$distro ${image_base[$distro]} \
    && echo "image_build_succesful" >> $log_file \
    || echo "image_build_failed" >> $log_file
  echo "Setting up image for: $distro" >> $log_file
  sudo singularity exec --writable  -B ../../install/scripts:/mnt \
    $HIPSYCL_TEST_DIR/hipsycl-$distro sh /mnt/add-repo-$distro.sh /test_spack_build \
    && echo "add_distro_succesful" >> $log_file \
    || echo "add_distro_failed" >> $log_file
done
for HIPSYCL_WITH_CUDA in OFF ON ; do
for HIPSYCL_WITH_ROCM in OFF ON ; do
  dict_key=$HIPSYCL_WITH_ROCM$HIPSYCL_WITH_CUDA
  echo "Building Package with ROCM $HIPSYCL_WITH_ROCM CUDA $HIPSYCL_WITH_CUDA pkg_suffix: ${pkg_suffix[$dict_key]}" >> $log_file
  export HIPSYCL_WITH_CUDA
  export HIPSYCL_WITH_ROCM 
  for distro in ${distros[@]}; do
    echo "Setting up image for: $distro" >> $log_file
    if [ "$HIPSYCL_WITH_CUDA" = "ON" ]; then
       sudo singularity exec --writable  -B ../../install/scripts:/mnt \
         $HIPSYCL_TEST_DIR/hipsycl-$distro sh /mnt/install-cuda.sh \
         && echo "cuda_install_succesful" >> $log_file \
         || echo "cuda_install_failed" >> $log_file
    else
      echo "cuda_install_skipped" >> $log_file
    fi
    sudo singularity exec --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro \
      ${install_cmd[$distro]}${pkg_suffix[$dict_key]}$HIPSYCL_PKG_TYPE \
      > $HIPSYCL_TEST_LOG_DIR/out \
      && echo "pkg_install_succesful" >> $log_file \
      || echo "pkg_install_failed" >> $log_file 
    echo "Depencies pulled in: " >> $log_file
    grep -i hipsycl $HIPSYCL_TEST_LOG_DIR/out >> $log_file
  done 
  echo "Start testing" >> $log_file
  HIPSYCL_PKG_CONTAINER_DIR=$HIPSYCL_TEST_DIR `pwd`/test_installation.sh 
  for distro in ${distros[@]}; do
    sudo singularity exec --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro \
      ${cleanup_cmd[$distro]}${pkg_suffix[$dict_key]}$HIPSYCL_PKG_TYPE\
      && echo "$distro pkg_cleanup_succesful" >> $log_file \
      || echo "$distro pkg_cleanup_failed" >> $log_file
  
    sudo singularity exec --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro \
      ${cleanup_dep[$distro]} \
      && echo "$distro pkg_dep_cleanup_succesful" >> $log_file \
      || echo "$distro pkg_dep_cleanup_failed" >> $log_file

     sudo singularity exec --writable  $HIPSYCL_TEST_DIR/hipsycl-$distro rm -rf /opt/hipSYCL/cuda
  done
done
done
