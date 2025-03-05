ARG BASE="ubuntu:22.04"

FROM ${BASE}

ARG LLVM_VERSION=18
# make empty to skip installation of ROCM and Intel Stack
ARG ROCM_VERSION="6.2.4"
# make empty to skip installation of CUDA
ARG CUDA_MAJOR_VERSION="11"
ARG CUDA_MINOR_VERSION="0"
ARG CUDA_PATCH_VERSION="2"
ARG CUDA_DRIVER_VERSION="450.51.05"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -u update \
    && apt-get -qq upgrade \
    # Setup Kitware repo for the latest cmake available:
    && apt-get -qq install \
        apt-transport-https ca-certificates gnupg software-properties-common wget apt-utils lsb-release \
    && wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null \
    && apt-add-repository 'deb https://apt.kitware.com/ubuntu/ jammy main' \
    && apt-get -u update \
    && apt-get -qq upgrade \
    && apt-get -qq install cmake \
        ca-certificates \
        build-essential \
        python3 \
        python3-pip \
        ninja-build \
        ccache \
        xz-utils \
        curl \
        git \
        bzip2 \
        lzma \
        xz-utils \
        unzip \
        libnuma-dev \
        gettext \
        jq \
        libtbb-dev \
        libboost-all-dev \
        ocl-icd-opencl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN <<EOF
    # skip if ROCM_VERSION empty (yeah.. Intel not ROCm .. but whatever)
    if [ -z "${ROCM_VERSION}" ]; then exit 0; fi
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16695.4/intel-igc-core_1.0.16695.4_amd64.deb
    wget -q https://github.com/intel/intel-graphics-compiler/releases/download/igc-1.0.16695.4/intel-igc-opencl_1.0.16695.4_amd64.deb
    wget -q https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-level-zero-gpu-dbgsym_1.3.29377.6_amd64.ddeb
    wget -q https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-level-zero-gpu_1.3.29377.6_amd64.deb
    wget -q https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-opencl-icd-dbgsym_24.17.29377.6_amd64.ddeb
    wget -q https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/intel-opencl-icd_24.17.29377.6_amd64.deb
    wget -q https://github.com/intel/compute-runtime/releases/download/24.17.29377.6/libigdgmm12_22.3.19_amd64.deb
    wget -q https://github.com/oneapi-src/level-zero/releases/download/v1.13.5/level-zero-devel_1.13.5+u22.04_amd64.deb
    wget -q https://github.com/oneapi-src/level-zero/releases/download/v1.13.5/level-zero_1.13.5+u22.04_amd64.deb
    dpkg -i *.deb
    rm *.deb
EOF

RUN <<EOF
    # skip if CUDA_MAJOR_VERSION empty
    if [ -z "${CUDA_MAJOR_VERSION}" ]; then exit 0; fi
    mkdir -p /opt/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}
    wget -q -O cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.sh https://developer.download.nvidia.com/compute/cuda/${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION}/local_installers/cuda_${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION}_${CUDA_DRIVER_VERSION}_linux.run
    sh ./cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.sh --override --silent --toolkit --no-man-page --no-drm --no-opengl-libs --installpath=/opt/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} && rm ./cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.sh
    ln -s /opt/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} /opt/cuda
    echo "CUDA Version ${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}.${CUDA_PATCH_VERSION}" | tee /opt/cuda-${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}/version.txt
EOF

RUN <<EOF
    # skip if ROCM_VERSION empty
    if [ -z "${ROCM_VERSION}" ]; then exit 0; fi
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add -
    echo "deb [arch=amd64] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} jammy main" | tee /etc/apt/sources.list.d/rocm.list
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600' | tee /etc/apt/preferences.d/rocm-pin-600
    apt-get update
    apt-get install -y rocm-dev
    rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
    wget -q https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    ./llvm.sh ${LLVM_VERSION}
    apt-get install -y libclang-${LLVM_VERSION}-dev clang-tools-${LLVM_VERSION} libomp-${LLVM_VERSION}-dev llvm-${LLVM_VERSION}-dev
    apt-get install -y -o DPkg::options::="--force-overwrite" libclang-rt-${LLVM_VERSION}-dev
    rm -rf /var/lib/apt/lists/*
    python3 -m pip install lit
    ln -s /usr/bin/FileCheck-${LLVM_VERSION} /usr/bin/FileCheck
EOF

ENV PATH="$PATH:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"
ENV CC=clang-${LLVM_VERSION}
ENV CXX=clang++-${LLVM_VERSION}


