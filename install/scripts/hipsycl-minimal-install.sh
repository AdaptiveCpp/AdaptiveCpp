#!/bin/bash
set -e

echo "This will install hipSYCL into the current directory in a VERY minimal configuration:"
echo "The installation will only support CPU and no LLVM compiler acceleration of SYCL kernels."
echo "For production use and performance, this may not be ideal, but if you just quickly want to have a SYCL implementation, it might be perfect :-)"
echo ""
echo "The only dependencies required are:"
echo " * Your default system compiler must support C++17 and OpenMP"
echo " * You need to have installed the boost.context and boost.fiber libraries, including development files (e.g. on Ubuntu, the libboost-all-dev package)."
echo " * python 3"
echo " * cmake"
echo ""
echo "Make sure these dependencies are satisfied and press enter to continue".
read ARG


rm -rf ./hipsycl-build
mkdir -p ./hipsycl-build
git clone https://github.com/illuhad/hipSYCL ./hipsycl-build
mkdir -p ./hipsycl-build/build
cd ./hipsycl-build/build
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/../.. -DWITH_CUDA_BACKEND=OFF -DWITH_ROCM_BACKEND=OFF -DWITH_LEVEL_ZERO_BACKEND=OFF -DWITH_OPENCL_BACKEND=OFF -DWITH_ACCELERATED_CPU=OFF -DWITH_SSCP_COMPILER=OFF ..
make install
