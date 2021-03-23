#!/bin/bash
set -e
set -o xtrace
if [ "$#" -lt 1 ]; then
  echo "
  Responsible for creating the packages from the built containers where the hipSYCL stack is installed.
  Currently there are two modes supported hipsycl and base. hipsycl will only build the base packages, base will build all the 
  base packages, (rocm, base,)
  Usage: <distro> <mode>
  "
  exit -1
fi
distro=$1
option=${2:-"hipsycl"}


HIPSYCL_PKG_BUILD_ROCM=ON
HIPSYCL_PKG_BUILD_BASE=ON
HIPSYCL_PKG_BUILD_HIPSYCL=ON

if [ "$option" = "hipsycl" ]; then
  HIPSYCL_PKG_BUILD_BASE=OFF 
  HIPSYCL_PKG_BUILD_ROCM=OFF
elif [ "$option" = "base" ]; then
  HIPSYCL_PKG_BUILD_HIPSYCL=OFF
fi

export HIPSYCL_PKG_BUILD_ROCM
export HIPSYCL_PKG_BUILD_BASE
export HIPSYCL_PKG_BUILD_HIPSYCL

source $HIPSYCL_PKG_DEVOPS_DIR/common/init.sh

HIPSYCL_PKG_DEVOPS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
HIPSYCL_PKG_SCRIPT_DIR=${HIPSYCL_PKG_SCRIPT_DIR:-../../install/scripts/}
HIPSYCL_PKG_SCRIPT_DIR_ABS=$HIPSYCL_PKG_DEVOPS_DIR/$HIPSYCL_PKG_SCRIPT_DIR
export HIPSYCL_PACKAGING_DIR="/tmp/hipsycl-packages-$distro"
cd $HIPSYCL_PKG_SCRIPT_DIR_ABS/packaging

export SINGULARITYENV_HIPSYCL_PACKAGING_DIR=$HIPSYCL_PACKAGING_DIR

stage_dir=${stage_dir[$distro]}

singularity exec $HIPSYCL_PKG_CONTAINER_DIR/hipsycl-${packaging_image[$distro]} bash ${packaging_script[$distro]}

mkdir -p $HIPSYCL_PKG_DEVOPS_DIR/$stage_dir
for file in `find /tmp/hipsycl-packages-$distro | grep ${find_built_pkg[$distro]}`; do
  mv $file $HIPSYCL_PKG_DEVOPS_DIR/$stage_dir/
done
rm -rf $SINGULARITYENV_HIPSYCL_PACKAGING_DIR