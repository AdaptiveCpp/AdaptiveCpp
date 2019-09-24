#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh


RPM_ROOT=${BUILD_DIR}/rpm
mkdir -p ${RPM_ROOT}/{SOURCES,BUILD,RPMS,SPECS,SRPMS,tmp}

rm -rf ${CUDA_DIR}/*
INSTALL_PREFIX=${CUDA_DIR}/opt/hipSYCL sh ../install-cuda.sh
rm -rf ${CUDA_DIR}/opt/hipSYCL/cuda/samples

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-cuda.spec
Summary: CUDA stack for hipSYCL
Name: hipSYCL-cuda
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: NVIDIA CUDA EULA
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-cuda-${HIPSYCL_VERSION_STRING}
AutoReq: no

%description
%{summary}

%install
cp -R ${CUDA_DIR}/* %{buildroot}


%files
/opt/hipSYCL/cuda

EOF


cd ${RPM_ROOT}/SPECS
rpmbuild -D '%_python_bytecompile_errors_terminate_build 0' -bb hipSYCL-cuda.spec

