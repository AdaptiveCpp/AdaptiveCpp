#!/bin/bash
set -e

HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-../../install/scripts/}
HIPSYCL_PKG_SCRIPT_DIR_ABS=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_SCRIPT_DIR
source $HIPSYCL_PKG_DEVOPS_DIR/common/init.sh

cd $HIPSYCL_PKG_SCRIPT_DIR_ABS/packaging

stage_dir=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_UBUNTU_PKG_DIR
echo "Starting Ubuntu 18.04 pkg build"
singularity exec $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-ubuntu-18.04 sh make-ubuntu-pkg.sh
echo "Moving to staging dir  $stage_dir"
mkdir -p $stage_dir
find /tmp/hipsycl-packages/ -name \*.deb -exec mv {} $HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_UBUNTU_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/

stage_dir=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_ARCH_PKG_DIR
echo "Starting arch build"
singularity exec $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-archlinux-rolling sh make-archlinux-pkg.sh
echo "Moving to staging dir  $stage_dir"
mkdir -p $stage_dir
find /tmp/hipsycl-packages/ -name \*.pkg.tar.zst -exec mv {} $HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_ARCH_PKG_DIR/ \;
find /tmp/hipsycl-packages/ -name \*.pkg.tar.zst.sig -exec mv {} $HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_ARCH_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/

stage_dir=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_CENTOS_PKG_DIR
echo "Starting Centos build"
singularity exec $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-centos-7 sh make-centos-pkg.sh
echo "Moving to stage dir $stage_dir"
mkdir -p $stage_dir
find /tmp/hipsycl-packages/ -name \*.rpm -exec mv {} $HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_CENTOS_PKG_DIR/ \;
rm -rf /tmp/hipsycl-packages/
