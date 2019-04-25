FROM nvidia/cuda:10.0-devel-ubuntu18.04

ARG hipsycl_branch=master
ARG hipsycl_origin=https://github.com/illuhad/hipSYCL
ENV hipsycl_branch=$hipsycl_branch
ENV hipsycl_origin=$hipsycl_origin
RUN apt-get update
RUN apt-get install -y wget
RUN echo "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main" >> /etc/apt/sources.list.d/llvm.list
RUN echo "deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main" >> /etc/apt/sources.list.d/llvm.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update
RUN apt-get install -y git llvm-6.0-dev llvm-6.0 clang-6.0 cmake libboost-all-dev clang-6.0-dev python3 libclang-6.0-dev
RUN apt-get install -y libllvm8 llvm-8 llvm-8-dev llvm-8-runtime clang-8 clang-tools-8 libclang-common-8-dev libclang-8-dev libclang1-8 libomp-8-dev
WORKDIR /tmp
RUN git clone -b $hipsycl_branch --recurse-submodules $hipsycl_origin
RUN mkdir /tmp/build
WORKDIR /tmp/build
ENV CXX=clang++-8
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LIBRARY_PATH=/usr/local/cuda/lib64:$LIBRARY_PATH
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=ON /tmp/hipSYCL
RUN make -j$(($(nproc) -1)) install
