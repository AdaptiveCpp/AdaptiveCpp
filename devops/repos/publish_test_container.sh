#!/bin/bash
set -o xtrace
set -e
distro=$1
cd $HIPSYCL_PKG_DEVOPS_DIR
commit_hash=`git rev-parse --short HEAD`
cd -
date=`date -u +"%Y%m%d"`
supported_backends="omp"
HIPSYCL_PKG_PUBLIC_CONTAINER_DIR=${HIPSYCL_PKG_PUBLIC_CONTAINER_DIR:-/data/repos/singularity/}
HIPSYCL_TEST_DIR=${HIPSYCL_TEST_DIR:-"/data/hipsyclbot/test-dir"}
mkdir -p $HIPSYCL_PKG_PUBLIC_CONTAINER_DIR
for backend in `ls $HIPSYCL_TEST_DIR | sed -n -e "s/^.*$distro-//p"`; do
    if [[ ${backend:0:1} = "1" ]]; then supported_backends="${supported_backends}-rocm"; fi
    if [[ ${backend:1:2} = "1" ]]; then supported_backends="${supported_backends}-cuda"; fi
    container_name_base="hipSYCL-${HIPSYCL_PKG_TYPE}-${distro}-${supported_backends}"
    container_name="${container_name_base}-${date}-${commit_hash}.sif"
    singularity exec --fakeroot --writable $HIPSYCL_TEST_DIR/hipsycl-$distro-$backend rm -rf /opt/hipSYCL/cuda
    #On arch these sockets cause a error while packing the container
    singularity exec --fakeroot --writable $HIPSYCL_TEST_DIR/hipsycl-$distro-$backend rm -rf /etc/pacman.d/gnupg/S.gpg-agent.browser \
            /etc/pacman.d/gnupg/S.gpg-agent.ssh /etc/pacman.d/gnupg/S.gpg-agent.extra  /etc/pacman.d/gnupg/S.gpg-agent
    singularity build --force --fakeroot $HIPSYCL_PKG_PUBLIC_CONTAINER_DIR/$container_name $HIPSYCL_TEST_DIR/hipsycl-$distro-$backend
    supported_backends="omp"
    #Keep only the two latest container from each kind
    ls -t $HIPSYCL_PKG_PUBLIC_CONTAINER_DIR | grep $container_name_base-[0-9] | tail -n +3 | xargs -I '_' rm -rf $HIPSYCL_PKG_PUBLIC_CONTAINER_DIR/_
done
