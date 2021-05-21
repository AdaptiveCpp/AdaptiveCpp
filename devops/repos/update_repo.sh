#!/bin/bash

set -e
set -o xtrace
if [ $1 = "--help" ]; then
  echo " 
   This file is responsible for driving the packaging, building, and testing process for the hipSYCL packaging system.
   It sets and exports defaults for the important environment variables that might concern the builds 
   
   Usage: $ update_repo.sh <distro> <action> [option]
   distro: centos-7, ubuntu-18.04 etc...
   action: build_base, build_hipsycl, package, deploy, test" 
  exit -1
fi

distro=$1
action=$2
option=$3
set +e
source /etc/profile
set -e
source ${HIPSYCL_PKG_ENV_FILE:-~/envs.out}

HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-../../install/scripts/}
HIPSYCL_PKG_SCRIPT_DIR_ABS=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_SCRIPT_DIR
HIPSYCL_PKG_REPO_BASE_DIR=${HIPSYCL_PKG_REPO_BASE_DIR:-/data/repos/}
HIPSYCL_PKG_REPO_BASE_DIR=$HIPSYCL_PKG_REPO_BASE_DIR/$HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
HIPSYCL_PKG_PUBLIC_CONTAINER_DIR=${HIPSYCL_PKG_PUBLIC_CONTAINER_DIR:-/data/repos/singularity/}
source $HIPSYCL_PKG_DEVOPS_DIR/common/init.sh

HIPSYCL_TEST_DIR="/data/hipsyclbot/test-dir"
mkdir -p $HIPSYCL_TEST_DIR

HIPSYCL_REPO_USER=${HIPSYCL_REPO_USER:-illuhad}
HIPSYCL_REPO_BRANCH=${HIPSYCL_REPO_BRANCH:-stable}

HIPSYCL_PKG_LLVM_VERSION_MAJOR=${HIPSYCL_PKG_LLVM_VERSION_MAJOR:-9}
HIPSYCL_PKG_LLVM_VERSION_MINOR=${HIPSYCL_PKG_LLVM_VERSION_MINOR:-0}
HIPSYCL_PKG_LLVM_VERSION_PATCH=${HIPSYCL_PKG_LLVM_VERSION_PATCH:-1}

HIPSYCL_HIP_VERSION=${HIPSYCL_HIP_VERSION:-4.0.0}

HIPSYCL_PKG_CONTAINER_DIR_SUFFIX=${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX:-containers}
HIPSYCL_PKG_CONTAINER_DIR_SUFFIX=${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX}${HIPSYCL_PKG_NAME_SUFFIX}
HIPSYCL_PKG_CONTAINER_DIR_NAME=${HIPSYCL_PKG_LLVM_REPO_BRANCH/release\//llvm-}-
HIPSYCL_PKG_CONTAINER_DIR=${HIPSYCL_PKG_CONTAINER_DIR:-$HIPSYCL_PKG_SCRIPT_DIR_ABS/${HIPSYCL_PKG_CONTAINER_DIR_NAME}-${HIPSYCL_PKG_CONTAINER_DIR_SUFFIX}}
HIPSYCL_PKG_TYPE=${HIPSYCL_PKG_TYPE:-nightly}

export HIPSYCL_PKG_CONTAINER_DIR
export HIPSYCL_PKG_LLVM_REPO_BRANCH
export HIPSYCL_PKG_LLVM_VERSION_MAJOR
export HIPSYCL_PKG_LLVM_VERSION_MINOR
export HIPSYCL_PKG_LLVM_VERSION_PATCH
export HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
export HIPSYCL_REPO_USER
export HIPSYCL_REPO_BRANCH
export HIPSYCL_PKG_TYPE
export HIPSYCL_PKG_NAME_SUFFIX
export HIPSYCL_PKG_DEVOPS_DIR
export HIPSYCL_WITH_CUDA
export HIPSYCL_WITH_ROCM


if [ "$action" = "build_base" ]; then
  bash $HIPSYCL_PKG_SCRIPT_DIR_ABS/rebuild-images.sh $distro $option
fi

if [ "$action" = "build_hipsycl" ]; then
  bash $HIPSYCL_PKG_SCRIPT_DIR_ABS/rebuild-images.sh $distro cleanup
  bash $HIPSYCL_PKG_SCRIPT_DIR_ABS/rebuild-images.sh $distro $option
fi

if [ "$action" = "package" ]; then
  bash $HIPSYCL_PKG_DEVOPS_DIR/create_pkgs.sh $distro $option
fi

if [ "$action" = "deploy" ]; then 
  bash $HIPSYCL_PKG_DEVOPS_DIR/create_repos.sh $distro
fi

if [ "$action" = "test" ]; then
  if [ -z "${@:4}" ]; then
    bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh $HIPSYCL_PKG_DEVOPS_DIR $distro $backend $option build $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
    bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh $HIPSYCL_PKG_DEVOPS_DIR $distro $backend $option add_repo $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
    bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh $HIPSYCL_PKG_DEVOPS_DIR $distro $backend $option install_dep $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX

    bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh $HIPSYCL_PKG_DEVOPS_DIR $distro $backend $option run_tests $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
    rm -rf /data/sbalint/singularity_tmp/*
  else
    bash $HIPSYCL_PKG_DEVOPS_DIR/test-packages.sh $HIPSYCL_PKG_DEVOPS_DIR $distro $backend $option $4 $HIPSYCL_PKG_REPO_BASE_DIR_SUFFIX
  fi
fi


if [ "$action" = "pub_cont" ]; then
   bash $HIPSYCL_PKG_DEVOPS_DIR/publish_test_container.sh $distro 
fi
