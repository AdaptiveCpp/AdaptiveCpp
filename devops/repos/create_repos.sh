#!/bin/bash
SINGULARITY_BASE_DIR=${SINGULARITY_BASE_DIR:-./containers/}
REPO_BASE_DIR=${REPO_BASE_DIR:-/data/repos/}

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/centos-7.sif sh create_centos_repo.sh

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
	 $SINGULARITY_BASE_DIR/ubuntu-18.04.sif sh create_ubuntu_repo.sh

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
	 $SINGULARITY_BASE_DIR/arch.sif sh create_arch_repo.sh
