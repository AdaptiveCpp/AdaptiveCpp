#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh

HIPSYCL_PKG_BUILD_BASE=${HIPSYCL_PKG_BUILD_BASE:-ON}
HIPSYCL_PKG_BUILD_HIPSYCL=${HIPSYCL_PKG_BUILD_HIPSYCL:-ON}
HIPSYCL_PKG_BUILD_ROCM=${HIPSYCL_PKG_BUILD_ROCM:-ON}
HIPSYCL_PKG_BUILD_CUDA=${HIPSYCL_PKG_BUILD_CUDA:-OFF}

RPM_ROOT=${BUILD_DIR}/rpm
mkdir -p ${RPM_ROOT}/{SOURCES,BUILD,RPMS,SPECS,SRPMS,tmp}


cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-core-${HIPSYCL_PKG_TYPE}.spec
Summary: Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs
Name: hipSYCL-core-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-omp-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_CORE_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/bin
/opt/hipSYCL/lib
/opt/hipSYCL/include
/opt/hipSYCL/etc

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-cuda-${HIPSYCL_PKG_TYPE}.spec
Summary: cuda backend for hipSYCL
Name: hipSYCL-cuda-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-core-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_CUDA_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/lib/hipSYCL

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-rocm-${HIPSYCL_PKG_TYPE}.spec
Summary: rocm backend for hipSYCL
Name: hipSYCL-rocm-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-base-rocm-${HIPSYCL_PKG_TYPE}, hipSYCL-core-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_ROCM_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/lib/hipSYCL

EOF
cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-omp-${HIPSYCL_PKG_TYPE}.spec
Summary: omp backend for hipSYCL
Name: hipSYCL-omp-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_PKG_TYPE}-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-base-${HIPSYCL_PKG_TYPE}, hipSYCL-core-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_OMP_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/lib/hipSYCL
EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-${HIPSYCL_PKG_TYPE}.spec
Summary: Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs
Name: hipSYCL-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_VERSION_STRING}
Requires: hipSYCL-rocm-${HIPSYCL_PKG_TYPE},  hipSYCL-cuda-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install

%global __python %{__python3}

%files

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-full-${HIPSYCL_PKG_TYPE}.spec
Summary: Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs
Name: hipSYCL-full-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_VERSION_STRING}
Requires: hipSYCL-${HIPSYCL_PKG_TYPE}
AutoReq: no

%description
%{summary}

%install

%global __python %{__python3}

%files

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-base.spec
Summary: base LLVM compiler stack for hipSYCL
Name: hipSYCL-base-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-base-${HIPSYCL_VERSION_STRING}
Requires: devtoolset-9, binutils, lbzip2
AutoReq: no

%description
%{summary}

%install
cp -R ${COMMON_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/llvm
/opt/hipSYCL/boost

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-base-rocm.spec
Summary: ROCm stack for hipSYCL
Name: hipSYCL-base-rocm-${HIPSYCL_PKG_TYPE}
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-rocm-${HIPSYCL_VERSION_STRING}
Requires: numactl-devel, numactl-libs, pciutils-devel, pciutils-libs, perl, elfutils-libelf-devel
AutoReq: no

%description
%{summary}

%install
cp -R ${ROCM_DIR}/* %{buildroot}

%global __python %{__python3}

%files
/opt/hipSYCL/rocm

EOF

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

%global __python %{__python3}

%files
/opt/hipSYCL/cuda

EOF


cd ${RPM_ROOT}/SPECS

if [ "$HIPSYCL_PKG_BUILD_HIPSYCL" = "ON"  ]; then
rpmbuild -bb hipSYCL-${HIPSYCL_PKG_TYPE}.spec
rpmbuild -bb hipSYCL-full-${HIPSYCL_PKG_TYPE}.spec

rpmbuild -bb hipSYCL-core-${HIPSYCL_PKG_TYPE}.spec
rpmbuild -bb hipSYCL-cuda-${HIPSYCL_PKG_TYPE}.spec
rpmbuild -bb hipSYCL-rocm-${HIPSYCL_PKG_TYPE}.spec
rpmbuild -bb hipSYCL-omp-${HIPSYCL_PKG_TYPE}.spec
fi

if [ "$HIPSYCL_PKG_BUILD_BASE" = "ON"  ]; then
rpmbuild -bb hipSYCL-base.spec
fi

if [ "$HIPSYCL_PKG_BUILD_ROCM" = "ON"  ]; then
rpmbuild -bb hipSYCL-base-rocm.spec
fi

if [ "$HIPSYCL_PKG_BUILD_CUDA" = "ON"  ]; then
rpmbuild -D '%_python_bytecompile_errors_terminate_build 0' -bb hipSYCL-cuda.spec
fi
