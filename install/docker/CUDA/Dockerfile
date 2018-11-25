FROM nvidia/cuda:10.0-devel-ubuntu18.04

RUN apt-get update
RUN apt-get install -y git llvm-6.0-dev llvm-6.0 clang-6.0 cmake libboost-dev clang-6.0-dev python3 libclang-6.0-dev
WORKDIR /tmp
RUN git clone --recurse-submodules https://github.com/illuhad/hipSYCL
RUN mkdir /tmp/build
WORKDIR /tmp/build
ENV CXX=clang++-6.0
ENV PATH=/usr/local/cuda/bin:$PATH
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr /tmp/hipSYCL
RUN make install
