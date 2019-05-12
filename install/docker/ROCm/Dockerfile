FROM rocm/rocm-terminal

ARG hipsycl_branch=master
ARG hipsycl_origin=https://github.com/illuhad/hipSYCL
ENV hipsycl_branch=$hipsycl_branch
ENV hipsycl_origin=$hipsycl_origin
RUN sudo apt-get update
RUN sudo apt-get install -y python3 libclang-6.0-dev clang-6.0 llvm-6.0-dev libboost-all-dev gcc
WORKDIR /tmp
RUN git clone -b $hipsycl_branch --recurse-submodules $hipsycl_origin
RUN mkdir /tmp/build
WORKDIR /tmp/build
USER root
ENV CXX=clang++-6.0
ENV PATH=/opt/rocm/bin:$PATH
ENV HIPSYCL_GPU_ARCH=gfx900
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DWITH_CPU_BACKEND=ON -DWITH_ROCM_BACKEND=ON /tmp/hipSYCL
RUN make -j$(($(nproc) -1)) install
