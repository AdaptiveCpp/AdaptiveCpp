#!/bin/bash
set -o xtrace
distro=$1
if [ -z $1 ]; then
  echo "Provide the name of the distro as the first command line argument"
  exit -1
fi

HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source $HIPSYCL_PKG_DEVOPS_DIR/common/init.sh

cd $HIPSYCL_PKG_DEVOPS_DIR
SINGULARITY_BASE_DIR=${SINGULARITY_BASE_DIR:-./containers}
HIPSYCL_PKG_REPO_BASE_DIR=${HIPSYCL_PKG_REPO_BASE_DIR:-/data/repos}
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-$HIPSYCL_PKG_DEVOPS_DIR/repo-creation-scripts}
mkdir -p $HIPSYCL_PKG_REPO_BASE_DIR

echo "$HIPSYCL_PKG_REPO_BASE_DIR"
singularity -d exec --fakeroot -B $HIPSYCL_PKG_REPO_BASE_DIR:/data/repos/ -B $HIPSYCL_PKG_DEVOPS_DIR:$HIPSYCL_PKG_DEVOPS_DIR \
     $SINGULARITY_BASE_DIR/${repo_tools_cont[$distro]} bash $HIPSYCL_PKG_SCRIPT_DIR/${repo_script[$distro]}
