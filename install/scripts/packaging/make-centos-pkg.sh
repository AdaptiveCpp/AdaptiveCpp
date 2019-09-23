#!/bin/bash
# Intended to be executed inside the built singularity container

set -e

. ./common/init.sh


RPM_ROOT=${BUILD_DIR}/rpm
mkdir -p ${RPM_ROOT}/{SOURCES,BUILD,RPMS,SPECS,SRPMS,tmp}


cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL.spec
Summary: Implementation of Khronos SYCL for CPUs, AMD GPUs and NVIDIA GPUs
Name: hipSYCL
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: BSD
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-${HIPSYCL_VERSION_STRING}
Requires: python3, hipSYCL-base
AutoReq: no

%description
%{summary}

%install
cp -R ${HIPSYCL_DIR}/* %{buildroot}

%files
/opt/hipSYCL/bin
/opt/hipSYCL/lib
/opt/hipSYCL/include
/opt/hipSYCL/etc

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-base.spec
Summary: base LLVM compiler stack for hipSYCL
Name: hipSYCL-base
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-base-${HIPSYCL_VERSION_STRING}
Requires: devtoolset-7

%description
%{summary}

%install
cp -R ${COMMON_DIR}/* %{buildroot}

%files
/opt/hipSYCL/llvm

EOF

cat << EOF > ${RPM_ROOT}/SPECS/hipSYCL-rocm.spec
Summary: ROCm stack for hipSYCL
Name: hipSYCL-rocm
Version: ${HIPSYCL_VERSION}
Release: ${HIPSYCL_BUILD}
License: LLVM
Packager: Aksel Alpay
Group: Development/Tools
BuildRequires: coreutils
BuildRoot: ${RPM_ROOT}/tmp/hipSYCL-rocm-${HIPSYCL_VERSION_STRING}
Requires: hipSYCL, numactl-devel, numactl-libs, pciutils-devel, pciutils-libs, perl, elfutils-libelf-devel

%description
%{summary}

%install
cp -R ${ROCM_DIR}/* %{buildroot}
  
%files
/opt/hipSYCL/rocm

EOF

cd ${RPM_ROOT}/SPECS
rpmbuild -bb hipSYCL.spec
rpmbuild -bb hipSYCL-base.spec
rpmbuild -bb hipSYCL-rocm.spec

