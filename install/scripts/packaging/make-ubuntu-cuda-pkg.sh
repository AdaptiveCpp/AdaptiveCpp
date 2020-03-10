#!/bin/bash

set -e

. ./common/init.sh

mkdir -p ${CUDA_DIR}/DEBIAN

cat << EOF > ${CUDA_DIR}/DEBIAN/control
Package: hipsycl-cuda
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Depends: hipSYCL
Description: CUDA stack for hipSYCL
 Provides CUDA toolkit for hipSYCL
EOF

INSTALL_PREFIX=${CUDA_DIR}/opt/hipSYCL sh ../install-cuda.sh

cd ${BUILD_DIR}
dpkg-deb --build ${CUDA_PKG}

