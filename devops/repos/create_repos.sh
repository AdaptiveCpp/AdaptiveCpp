#!/bin/bash
HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $HIPSYCL_PKG_DEVOPS_DIR
SINGULARITY_BASE_DIR=${SINGULARITY_BASE_DIR:-./containers}
HIPSYCL_PKG_REPO_BASE_DIR=${HIPSYCL_PKG_REPO_BASE_DIR:-/data/repos}
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-./scripts}

sudo singularity exec -B $HIPSYCL_PKG_REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/centos-7.sif sh $HIPSYCL_PKG_SCRIPT_DIR/create_centos_repo.sh

sudo singularity exec -B $HIPSYCL_PKG_REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/ubuntu-18.04.sif bash $HIPSYCL_PKG_SCRIPT_DIR/create_ubuntu_repo.sh

sudo singularity exec -B $HIPSYCL_PKG_REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/arch.sif sh $HIPSYCL_PKG_SCRIPT_DIR/create_arch_repo.sh
