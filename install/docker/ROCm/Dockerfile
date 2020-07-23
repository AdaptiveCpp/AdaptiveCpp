FROM rocm/rocm-terminal

ARG hipsycl_branch=stable
ARG hipsycl_origin=https://github.com/illuhad/hipSYCL
ENV hipsycl_branch=$hipsycl_branch
ENV hipsycl_origin=$hipsycl_origin
USER root
RUN apt-get update
RUN apt-get install -y python3 libboost-all-dev gcc wget git
RUN echo "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" >> /etc/apt/sources.list.d/llvm.list
RUN echo "deb-src http://apt.llvm.org/xenial/ llvm-toolchain-xenial-8 main" >> /etc/apt/sources.list.d/llvm.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get update
RUN apt-get install -y libllvm8 llvm-8 llvm-8-dev llvm-8-runtime clang-8 clang-tools-8 libclang-common-8-dev libclang-8-dev libclang1-8 libomp-8-dev
WORKDIR /tmp
RUN git clone -b $hipsycl_branch --recurse-submodules $hipsycl_origin
RUN mkdir /tmp/build
WORKDIR /tmp/build
ENV CXX=clang++-8
ENV PATH=/opt/rocm/bin:$PATH
ENV HIPSYCL_GPU_ARCH=gfx900
RUN cmake -DCMAKE_C_COMPILER=clang-8 -DCMAKE_CXX_COMPILER=clang++-8 -DCMAKE_INSTALL_PREFIX=/usr -DWITH_CPU_BACKEND=ON -DWITH_ROCM_BACKEND=ON /tmp/hipSYCL
RUN make -j$(($(nproc) -1)) install
