#!/bin/bash
BASE_DIR=`pwd`
SINGULARITY_BASE_DIR=${SINGULARITY_BASE_DIR:-./containers}
REPO_BASE_DIR=${REPO_BASE_DIR:-/data/repos}
SCRIPT_DIR=${SCRIPT_DIR:-./scripts}

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/centos-7.sif sh $SCRIPT_DIR/create_centos_repo.sh

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/ubuntu-18.04.sif bash $SCRIPT_DIR/create_ubuntu_repo.sh

sudo singularity exec -B $REPO_BASE_DIR:/data/repos/ \
     $SINGULARITY_BASE_DIR/arch.sif sh $SCRIPT_DIR/create_arch_repo.sh
