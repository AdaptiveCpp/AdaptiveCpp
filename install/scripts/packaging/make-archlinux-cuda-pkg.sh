#!/bin/bash

set -e

. ./common/init.sh

mkdir -p ${CUDA_DIR}/pkg
cp ../install-cuda.sh ${CUDA_DIR}/pkg/


cat << EOF > ${CUDA_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-cuda
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="CUDA stack for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('NVIDIA CUDA EULA')
depends=('hipSYCL')
source=('install-cuda.sh')
md5sums=()
validpgpkeys=()

package(){
  INSTALL_PREFIX=\$pkgdir/opt/hipSYCL sh ./install-cuda.sh
}

EOF

cd ${CUDA_DIR}/pkg && makepkg -d -c --skipinteg


