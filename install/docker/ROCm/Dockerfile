FROM rocm/rocm-terminal

RUN sudo apt-get update
RUN sudo apt-get install -y python3 libclang-6.0-dev clang-6.0 llvm-6.0-dev libboost-dev gcc
WORKDIR /tmp
RUN git clone --recurse-submodules https://github.com/illuhad/hipSYCL
RUN mkdir /tmp/build
WORKDIR /tmp/build
USER root
ENV CXX=clang++-6.0
ENV PATH=/opt/rocm/bin:$PATH
ENV HIPSYCL_GPU_ARCH=gfx900
RUN cmake -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_CXX_FLAGS="-std=c++14" -DWITH_CPU_BACKEND=ON -DWITH_ROCM_BACKEND=ON /tmp/hipSYCL
RUN make install

