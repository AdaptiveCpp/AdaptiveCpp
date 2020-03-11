#!/bin/bash

source ./common/init.sh

SCRIPT_DIR=${SCRIPT_DIR:-../../install/scripts/}
BASE_DIR=`pwd`
set -e

export BUILD_HIPSYCL=ON
export BUILD_CUDA=OFF
export BUILD_ROCM=OFF
export BUILD_BASE=OFF
export HIPSYCL_GPG_KEY=B2B75080

cd $SCRIPT_DIR

sh ./rebuild-hipsycl-images.sh

cd $SCRIPT_DIR/packaging

echo "Starting Ubuntu 18.04 pkg build"
singularity exec ../hipsycl-ubuntu-18.04.sif sh make-ubuntu-pkg.sh
echo "Moving to staging dir"
find /tmp/hipsycl-packages/ -name \*.deb -exec mv {} $BASE_DIR/$UBUNTU_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/

echo "Starting arch build"
singularity exec ../hipsycl-archlinux-rolling.sif sh make-archlinux-pkg.sh
echo "Moving to stage dir"
find /tmp/hipsycl-packages/ -name \*.pkg.tar.xz -exec mv {} $BASE_DIR/$ARCH_PKG_DIR/ \;
find /tmp/hipsycl-packages/ -name \*.pkg.tar.xz.sig -exec mv {} $BASE_DIR/$ARCH_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/

echo "Starting Centos build"
singularity exec ../hipsycl-centos-7.sif sh make-centos-pkg.sh
echo "Moving to stage dir"
find /tmp/hipsycl-packages/ -name \*.rpm -exec mv {} $BASE_DIR/$CENTOS_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/
