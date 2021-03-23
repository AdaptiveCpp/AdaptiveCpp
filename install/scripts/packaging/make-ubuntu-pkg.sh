#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh

HIPSYCL_PKG_BUILD_BASE=${HIPSYCL_PKG_BUILD_BASE:-ON}
HIPSYCL_PKG_BUILD_HIPSYCL=${HIPSYCL_PKG_BUILD_HIPSYCL:-ON}
HIPSYCL_PKG_BUILD_ROCM=${HIPSYCL_PKG_BUILD_ROCM:-ON}
HIPSYCL_PKG_BUILD_CUDA=${HIPSYCL_PKG_BUILD_CUDA:-OFF}


mkdir -p ${CUDA_DIR}/DEBIAN
mkdir -p ${ROCM_DIR}/DEBIAN
mkdir -p ${COMMON_DIR}/DEBIAN

mkdir -p ${HIPSYCL_CORE_DIR}/DEBIAN
mkdir -p ${HIPSYCL_CUDA_DIR}/DEBIAN
mkdir -p ${HIPSYCL_ROCM_DIR}/DEBIAN
mkdir -p ${HIPSYCL_OMP_DIR}/DEBIAN

mkdir -p ${HIPSYCL_META_DIR}/DEBIAN
mkdir -p ${HIPSYCL_FULL_DIR}/DEBIAN

cat << EOF > ${HIPSYCL_CORE_DIR}/DEBIAN/control 
Package: hipsycl-core-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-omp-${HIPSYCL_PKG_TYPE}, python3 (>= 3.0)
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL${HIPSYCL_VERSION_STRING}
 Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs 
EOF

cat << EOF > ${HIPSYCL_CUDA_DIR}/DEBIAN/control 
Package: hipsycl-cuda-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-core-${HIPSYCL_PKG_TYPE}
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL${HIPSYCL_VERSION_STRING}
  Cuda backend for hipSYCL
EOF

cat << EOF > ${HIPSYCL_ROCM_DIR}/DEBIAN/control 
Package: hipsycl-rocm-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-core-${HIPSYCL_PKG_TYPE}, hipsycl-base-rocm-${HIPSYCL_PKG_TYPE} , python3 (>= 3.0)
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL${HIPSYCL_VERSION_STRING}
  Rocm backend for hipSYCL
EOF

cat << EOF > ${HIPSYCL_OMP_DIR}/DEBIAN/control 
Package: hipsycl-omp-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-core-${HIPSYCL_PKG_TYPE}, hipsycl-base-${HIPSYCL_PKG_TYPE} , python3 (>= 3.0)
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL${HIPSYCL_VERSION_STRING}
  omp backend for hipSYCL
EOF

cat << EOF > ${COMMON_DIR}/DEBIAN/control
Package: hipsycl-base-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: g++-9, libnuma1, build-essential
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: hipSYCL base compiler stack
 Provides an LLVM compiler stack for hipSYCL
EOF

cat << EOF > ${ROCM_DIR}/DEBIAN/control
Package: hipsycl-base-rocm-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends:  libpci-dev
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: ROCm compiler stack for hipSYCL-${HIPSYCL_PKG_TYPE}  Provides ROCm libraries for hipSYCL
EOF


cat << EOF > ${HIPSYCL_FULL_DIR}/DEBIAN/control
Package: hipsycl-full-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-rocm-${HIPSYCL_PKG_TYPE},  hipsycl-cuda-${HIPSYCL_PKG_TYPE}
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description:  Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs 

EOF

cat << EOF > ${HIPSYCL_META_DIR}/DEBIAN/control
Package: hipsycl-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Depends: hipsycl-full-${HIPSYCL_PKG_TYPE}
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description:  Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs 

EOF

cat << EOF > ${CUDA_DIR}/DEBIAN/control
Package: hipsycl-base-cuda-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION_STRING}
Section: base
Priority: optional
Architecture: amd64
Maintainer: Aksel Alpay <aksel.alpay@uni-heidelberg.de>
Description: CUDA stack for hipSYCL
 Provides CUDA toolkit for hipSYCL
EOF

cd ${BUILD_DIR}

if [ "$HIPSYCL_PKG_BUILD_ROCM" = "ON" ]; then
dpkg-deb --build ${ROCM_PKG}
fi

if [ "$HIPSYCL_PKG_BUILD_BASE" = "ON"  ]; then
dpkg-deb --build ${COMMON_PKG}
fi

if [ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON" ]; then

dpkg-deb --build ${HIPSYCL_CORE_DIR} 
dpkg-deb --build ${HIPSYCL_CUDA_DIR} 
dpkg-deb --build ${HIPSYCL_ROCM_DIR} 
dpkg-deb --build ${HIPSYCL_OMP_DIR} 

dpkg-deb --build ${HIPSYCL_META_PKG}
dpkg-deb --build ${HIPSYCL_FULL_PKG}
fi

if [ "$HIPSYCL_PKG_BUILD_CUDA" = "ON" ]; then
dpkg-deb --build ${CUDA_PKG}
fi

