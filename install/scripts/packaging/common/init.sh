#!/bin/bash
# Intended to be executed inside the built singularity container

# define variables - version and build paths
HIPSYCL_VERSION=0.8.1
HIPSYCL_BUILD=`date +%Y%m%d`
HIPSYCL_VERSION_STRING=${HIPSYCL_VERSION}-${HIPSYCL_BUILD}
HIPSYCL_GPG_KEY=${HIPSYCL_GPG_KEY:-B2B75080}

#BUILD_DIR=`mktemp -d`
BUILD_DIR=${HIPSYCL_PACKAGING_DIR:-/tmp/hipsycl-packages}
CUDA_PKG=hipSYCL-cuda-${HIPSYCL_VERSION_STRING}
ROCM_PKG=hipSYCL-rocm-${HIPSYCL_VERSION_STRING}
COMMON_PKG=hipSYCL-base-${HIPSYCL_VERSION_STRING}
HIPSYCL_PKG=hipSYCL-${HIPSYCL_VERSION_STRING}

echo "Building packages in directory ${BUILD_DIR}..."

export CUDA_DIR=${BUILD_DIR}/${CUDA_PKG}
export ROCM_DIR=${BUILD_DIR}/${ROCM_PKG}
export COMMON_DIR=${BUILD_DIR}/${COMMON_PKG}
export HIPSYCL_DIR=${BUILD_DIR}/${HIPSYCL_PKG}

# Make sure there are no residual files
# from previous builds
rm -rf ${CUDA_DIR}/opt || true
rm -rf ${ROCM_DIR}/opt || true
rm -rf ${COMMON_DIR}/opt || true
rm -rf ${HIPSYCL_DIR}/opt || true

# create build directories
mkdir -p ${CUDA_DIR}/opt/hipSYCL/cuda
mkdir -p ${ROCM_DIR}/opt/hipSYCL/rocm
mkdir -p ${COMMON_DIR}/opt/hipSYCL/llvm
mkdir -p ${HIPSYCL_DIR}/opt/hipSYCL

# sort installed binaries into build paths
cp -R /opt/hipSYCL/rocm/* ${ROCM_DIR}/opt/hipSYCL/rocm || true
cp -R /opt/hipSYCL/llvm/* ${COMMON_DIR}/opt/hipSYCL/llvm || true

cp -R /opt/hipSYCL/bin     ${HIPSYCL_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/etc     ${HIPSYCL_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/include ${HIPSYCL_DIR}/opt/hipSYCL || true
cp -R /opt/hipSYCL/lib     ${HIPSYCL_DIR}/opt/hipSYCL || true

