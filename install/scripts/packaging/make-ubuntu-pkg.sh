#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh

BUILD_BASE=${BUILD_BASE:-ON}
BUILD_HIPSYCL=${BUILD_HIPSYCL:-ON}
BUILD_ROCM=${BUILD_ROCM:-ON}
BUILD_CUDA=${BUILD_CUDA:-OFF}

mkdir -p ${CUDA_DIR}/DEBIAN
mkdir -p ${ROCM_DIR}/DEBIAN
mkdir -p ${COMMON_DIR}/DEBIAN
mkdir -p ${HIPSYCL_DIR}/DEBIAN

cat << EOF > ${HIPSYCL_DIR}/DEBIAN/control 
Package: hipsycl
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-base (>= 0.8), python3 (>= 3.0)
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL
 Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs 
EOF

cat << EOF > ${COMMON_DIR}/DEBIAN/control
Package: hipsycl-base
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: g++, libnuma1
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL base compiler stack
 Provides an LLVM compiler stack for hipSYCL
EOF

cat << EOF > ${ROCM_DIR}/DEBIAN/control
Package: hipsycl-rocm
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl (>= 0.8), perl, perl-modules, libpci-dev
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: ROCm compiler stack for hipSYCL
 Provides ROCm libraries for hipSYCL
EOF

cat << EOF > ${CUDA_DIR}/DEBIAN/control
Package: hipsycl-cuda
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: CUDA stack for hipSYCL
 Provides CUDA toolkit for hipSYCL
EOF

cd ${BUILD_DIR}

if [ "$BUILD_ROCM" = "ON" ]; then
dpkg-deb --build ${ROCM_PKG}
fi

if [ "$BUILD_BASE" = "ON"  ]; then
dpkg-deb --build ${COMMON_PKG}
fi

if [ "$BUILD_HIPSYCL" = "ON" ]; then
dpkg-deb --build ${HIPSYCL_PKG}
fi

if [ "$BUILD_CUDA" = "ON" ]; then
dpkg-deb --build ${CUDA_PKG}
fi

