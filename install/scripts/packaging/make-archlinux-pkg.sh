#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh

HIPSYCL_PKG_BUILD_BASE=${HIPSYCL_PKG_BUILD_BASE:-ON}
HIPSYCL_PKG_BUILD_HIPSYCL=${HIPSYCL_PKG_BUILD_HIPSYCL:-ON}
HIPSYCL_PKG_BUILD_ROCM=${HIPSYCL_PKG_BUILD_ROCM:-ON}
HIPSYCL_PKG_BUILD_CUDA=${HIPSYCL_PKG_BUILD_CUDA:-OFF}

echo $HIPSYCL_GPG_KEY
if [ -n "$HIPSYCL_GPG_KEY" ]; then
	SIGN=" --sign --key $HIPSYCL_GPG_KEY"
fi

tar -cvf ${BUILD_DIR}/cuda-pkg.tar.gz -C ${CUDA_DIR} opt/
tar -cvf ${BUILD_DIR}/rocm-pkg.tar.gz -C ${ROCM_DIR} opt/
tar -cvf ${BUILD_DIR}/common-pkg.tar.gz -C ${COMMON_DIR} opt/
tar -cvf ${BUILD_DIR}/hipsycl-pkg.tar.gz -C ${HIPSYCL_DIR} opt/

mkdir -p ${CUDA_DIR}/pkg
mkdir -p ${ROCM_DIR}/pkg
mkdir -p ${COMMON_DIR}/pkg
mkdir -p ${HIPSYCL_DIR}/pkg

mv ${BUILD_DIR}/cuda-pkg.tar.gz ${CUDA_DIR}/pkg/
mv ${BUILD_DIR}/rocm-pkg.tar.gz ${ROCM_DIR}/pkg/
mv ${BUILD_DIR}/common-pkg.tar.gz ${COMMON_DIR}/pkg/
mv ${BUILD_DIR}/hipsycl-pkg.tar.gz ${HIPSYCL_DIR}/pkg/

cat << EOF > ${HIPSYCL_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('BSD')
depends=('hipSYCL-base${HIPSYCL_PKG_NAME_SUFFIX}' 'python' 'boost')
provides=('hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}' 'SYCL${HIPSYCL_PKG_NAME_SUFFIX}')
source=('hipsycl-pkg.tar.gz')
md5sums=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${COMMON_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-base${HIPSYCL_PKG_NAME_SUFFIX}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="LLVM compiler stack for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('LLVM')
depends=('numactl')
source=('common-pkg.tar.gz')
md5sums=()
validpgpkeys=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${ROCM_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-rocm${HIPSYCL_PKG_NAME_SUFFIX}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="ROCm compiler stack and libraries for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('LLVM')
depends=('hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}' 'pciutils' 'libelf' 'perl' 'pkg-config')
provides=('hipSYCL${HIPSYCL_PKG_NAME_SUFFIX}' 'SYCL${HIPSYCL_PKG_NAME_SUFFIX}')
source=('rocm-pkg.tar.gz')
md5sums=()
validpgpkeys=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF



cat << EOF > ${CUDA_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-cuda${HIPSYCL_PKG_NAME_SUFFIX}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="CUDA stack for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('NVIDIA CUDA EULA')
depends=()
provides=('cuda')
source=('cuda-pkg.tar.gz')
md5sums=()
validpgpkeys=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

if [ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON" ]; then
cd ${HIPSYCL_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
fi

if [ "$HIPSYCL_PKG_BUILD_BASE" = "ON" ]; then
cd ${COMMON_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
fi

if [ "$HIPSYCL_PKG_BUILD_ROCM" = "ON" ]; then
cd ${ROCM_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
fi

if [ "$HIPSYCL_PKG_BUILD_CUDA" = "ON" ]; then
cd ${CUDA_DIR}/pkg && makepkg -d -c --skipinteg $SIGN
echo $HIPSYCL_PKG_BUILD_CUDA
fi
