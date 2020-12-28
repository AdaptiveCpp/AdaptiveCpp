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

tar -cvf ${BUILD_DIR}/hipsycl-core-pkg.tar.gz -C ${HIPSYCL_CORE_DIR} opt/
tar -cvf ${BUILD_DIR}/hipsycl-cuda-pkg.tar.gz -C ${HIPSYCL_CUDA_DIR} opt/
tar -cvf ${BUILD_DIR}/hipsycl-rocm-pkg.tar.gz -C ${HIPSYCL_ROCM_DIR} opt/
tar -cvf ${BUILD_DIR}/hipsycl-omp-pkg.tar.gz -C ${HIPSYCL_OMP_DIR}  opt/

mkdir -p ${CUDA_DIR}/pkg
mkdir -p ${ROCM_DIR}/pkg
mkdir -p ${COMMON_DIR}/pkg

mkdir -p ${HIPSYCL_CORE_DIR}/pkg
mkdir -p ${HIPSYCL_CUDA_DIR}/pkg
mkdir -p ${HIPSYCL_ROCM_DIR}/pkg
mkdir -p ${HIPSYCL_OMP_DIR}/pkg

mkdir -p ${HIPSYCL_FULL_DIR}/pkg
mkdir -p ${HIPSYCL_META_DIR}/pkg

mv ${BUILD_DIR}/cuda-pkg.tar.gz ${CUDA_DIR}/pkg/
mv ${BUILD_DIR}/rocm-pkg.tar.gz ${ROCM_DIR}/pkg/
mv ${BUILD_DIR}/common-pkg.tar.gz ${COMMON_DIR}/pkg/

mv ${BUILD_DIR}/hipsycl-core-pkg.tar.gz ${HIPSYCL_CORE_DIR}/pkg
mv ${BUILD_DIR}/hipsycl-cuda-pkg.tar.gz ${HIPSYCL_CUDA_DIR}/pkg
mv ${BUILD_DIR}/hipsycl-rocm-pkg.tar.gz ${HIPSYCL_ROCM_DIR}/pkg
mv ${BUILD_DIR}/hipsycl-omp-pkg.tar.gz ${HIPSYCL_OMP_DIR}/pkg



cat << EOF > ${HIPSYCL_CORE_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-core-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('BSD')
depends=('hipSYCL-omp-${HIPSYCL_PKG_TYPE}' 'python' )
provides=('hipSYCL-core-${HIPSYCL_PKG_TYPE}' )
source=('hipsycl-core-pkg.tar.gz')
md5sums=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${HIPSYCL_CUDA_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-cuda-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="cuda backend for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('BSD')
depends=( 'hipSYCL-core-${HIPSYCL_PKG_TYPE}' )
provides=('hipSYCL-cuda-${HIPSYCL_PKG_TYPE}' )
source=('hipsycl-cuda-pkg.tar.gz')
md5sums=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${HIPSYCL_ROCM_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-rocm-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="rocm backend for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('BSD')
depends=('hipSYCL-base-rocm-${HIPSYCL_PKG_TYPE}' 'hipSYCL-core-${HIPSYCL_PKG_TYPE}' )
provides=('hipSYCL-rocm-${HIPSYCL_PKG_TYPE}' )
source=('hipsycl-rocm-pkg.tar.gz')
md5sums=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${HIPSYCL_OMP_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-omp-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="omp backend for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('BSD')
depends=('hipSYCL-base-${HIPSYCL_PKG_TYPE}' 'hipSYCL-core-${HIPSYCL_PKG_TYPE}'  )
provides=('hipSYCL-omp-${HIPSYCL_PKG_TYPE}' )
source=('hipsycl-omp-pkg.tar.gz')
md5sums=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${COMMON_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-base-${HIPSYCL_PKG_TYPE}
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
pkgname=hipSYCL-base-rocm-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="ROCm compiler stack and libraries for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('LLVM')
depends=( 'pciutils' 'libelf' 'perl' 'pkg-config')
provides=('hipSYCL-${HIPSYCL_PKG_TYPE}' 'SYCL-${HIPSYCL_PKG_TYPE}')
source=('rocm-pkg.tar.gz')
md5sums=()
validpgpkeys=()


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

cat << EOF > ${HIPSYCL_FULL_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-full-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('LLVM')
depends=( 'hipSYCL-${HIPSYCL_PKG_TYPE}' )
provides=( 'hipSYCL-full-${HIPSYCL_PKG_TYPE}' )

EOF

cat << EOF > ${HIPSYCL_META_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('LLVM')
depends=( 'hipSYCL-cuda-${HIPSYCL_PKG_TYPE}' 'hipSYCL-rocm-${HIPSYCL_PKG_TYPE}' 'hipSYCL-core-${HIPSYCL_PKG_TYPE}' )
provides=( 'hipSYCL-${HIPSYCL_PKG_TYPE}' )

EOF

cat << EOF > ${CUDA_DIR}/pkg/PKGBUILD
# Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
pkgname=hipSYCL-cuda-${HIPSYCL_PKG_TYPE}
pkgver=${HIPSYCL_VERSION}
pkgrel=${HIPSYCL_BUILD}
pkgdesc="CUDA stack for hipSYCL"
arch=('x86_64')
url="https://github.com/illuhad/hipSYCL"
license=('NVIDIA CUDA EULA')
depends=()
provides=('cuda')
source=('cuda-pkg.tar.gz')


package() {
  cp -R \$srcdir/opt \$pkgdir
}
EOF

if [ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON" ]; then
cd ${HIPSYCL_CORE_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
cd ${HIPSYCL_CUDA_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
cd ${HIPSYCL_ROCM_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
cd ${HIPSYCL_OMP_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN

cd ${HIPSYCL_META_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
cd ${HIPSYCL_FULL_DIR}/pkg && makepkg -d -c --skipinteg  $SIGN
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
