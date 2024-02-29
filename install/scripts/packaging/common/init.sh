#!/bin/bash
# Intended to be executed inside the built singularity container

# define variables - version and build paths
HIPSYCL_VERSION=24.02.0
HIPSYCL_BUILD=`date +%Y%m%d`
HIPSYCL_VERSION_STRING=${HIPSYCL_VERSION}-${HIPSYCL_BUILD}
HIPSYCL_GPG_KEY=${HIPSYCL_GPG_KEY:-B2B75080}

#BUILD_DIR=`mktemp -d`
BUILD_DIR=${HIPSYCL_PACKAGING_DIR:-/tmp/hipsycl-packages}

#Base packages
CUDA_PKG=hipSYCL-base-cuda-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
ROCM_PKG=hipSYCL-base-rocm-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
COMMON_PKG=hipSYCL-base-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}

#hipSYCL packages
HIPSYCL_CORE_PKG=hipSYCL-core-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
HIPSYCL_CUDA_PKG=hipSYCL-cuda-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
HIPSYCL_ROCM_PKG=hipSYCL-rocm-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
HIPSYCL_OMP_PKG=hipSYCL-omp-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}

#Meta packages
HIPSYCL_META_PKG=hipSYCL-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
HIPSYCL_FULL_PKG=hipSYCL-full-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}


echo "Building packages in directory ${BUILD_DIR}..."

export CUDA_DIR=${BUILD_DIR}/${CUDA_PKG}
export ROCM_DIR=${BUILD_DIR}/${ROCM_PKG}
export COMMON_DIR=${BUILD_DIR}/${COMMON_PKG}

export HIPSYCL_CORE_DIR=${BUILD_DIR}/${HIPSYCL_CORE_PKG}
export HIPSYCL_CUDA_DIR=${BUILD_DIR}/${HIPSYCL_CUDA_PKG}
export HIPSYCL_ROCM_DIR=${BUILD_DIR}/${HIPSYCL_ROCM_PKG}
export HIPSYCL_OMP_DIR=${BUILD_DIR}/${HIPSYCL_OMP_PKG}

export HIPSYCL_META_DIR=${BUILD_DIR}/${HIPSYCL_META_PKG}
export HIPSYCL_FULL_DIR=${BUILD_DIR}/${HIPSYCL_FULL_PKG}

# Make sure there are no residual files
# from previous builds
rm -rf ${CUDA_DIR}/opt || true
rm -rf ${ROCM_DIR}/opt || true
rm -rf ${COMMON_DIR}/opt || true

rm -rf ${HIPSYCL_CORE_DIR} || true
rm -rf ${HIPSYCL_CUDA_DIR} || true
rm -rf ${HIPSYCL_ROCM_DIR} || true
rm -rf ${HIPSYCL_OMP_DIR} || true

# create build directories
mkdir -p ${CUDA_DIR}/opt/hipSYCL/cuda
mkdir -p ${ROCM_DIR}/opt/hipSYCL/rocm
mkdir -p ${COMMON_DIR}/opt/hipSYCL

mkdir -p ${HIPSYCL_CORE_DIR}/opt/hipSYCL
mkdir -p ${HIPSYCL_CUDA_DIR}/opt/hipSYCL/lib/hipSYCL
mkdir -p ${HIPSYCL_ROCM_DIR}/opt/hipSYCL/lib/hipSYCL
mkdir -p ${HIPSYCL_OMP_DIR}/opt/hipSYCL/lib/hipSYCL


# sort installed binaries into build paths
cp -R /opt/hipSYCL/rocm/* ${ROCM_DIR}/opt/hipSYCL/rocm || true
cp -R /opt/hipSYCL/llvm ${COMMON_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/boost ${COMMON_DIR}/opt/hipSYCL || true

cp -R /opt/hipSYCL/bin     ${HIPSYCL_CORE_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/etc     ${HIPSYCL_CORE_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/include ${HIPSYCL_CORE_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/lib     ${HIPSYCL_CORE_DIR}/opt/hipSYCL || true
rm -rf ${HIPSYCL_CORE_DIR}/opt/hipSYCL/lib/hipSYCL/* || true

cp  /opt/hipSYCL/lib/hipSYCL/librt-backend-cuda.so ${HIPSYCL_CUDA_DIR}/opt/hipSYCL/lib/hipSYCL || true
cp  /opt/hipSYCL/lib/hipSYCL/librt-backend-hip.so ${HIPSYCL_ROCM_DIR}/opt/hipSYCL/lib/hipSYCL || true
cp  /opt/hipSYCL/lib/hipSYCL/librt-backend-omp.so ${HIPSYCL_OMP_DIR}/opt/hipSYCL/lib/hipSYCL || true

