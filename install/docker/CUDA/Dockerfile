FROM nvidia/cuda:10.0-devel-ubuntu18.04

ARG hipsycl_branch=master
ARG hipsycl_origin=https://github.com/illuhad/hipSYCL
ENV hipsycl_branch=$hipsycl_branch
ENV hipsycl_origin=$hipsycl_origin
RUN apt-get update
RUN apt-get install -y git llvm-6.0-dev llvm-6.0 clang-6.0 cmake libboost-dev clang-6.0-dev python3 libclang-6.0-dev
WORKDIR /tmp
RUN git clone -b $hipsycl_branch --recurse-submodules $hipsycl_origin
RUN mkdir /tmp/build
WORKDIR /tmp/build
ENV CXX=clang++-6.0
ENV PATH=/usr/local/cuda/bin:$PATH
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=ON /tmp/hipSYCL
RUN make install
