# Maintainer: Aksel Alpay <aksel.alpay at uni-heidelberg dot de>
pkgname=hipsycl-cuda-git
pkgver=0.7.9
pkgrel=1
pkgdesc="SYCL implementation over CUDA/HIP for NVIDIA devices."
arch=("x86_64")
url="https://github.com/illuhad/hipSYCL"
makedepends=("cmake")
provides=("hipSYCL" "SYCL")
license=("BSD")
depends=("cuda-9.2" "llvm" "clang" "boost" "openmp")

source=('hipSYCL::git+https://github.com/illuhad/hipSYCL.git')

md5sums=("SKIP")

pkgver() {
  cd "${srcdir}/hipSYCL"
  git log -1 --format=%cd.%h --date=short|tr -d -
}

prepare() {
  cd "${srcdir}/hipSYCL/contrib/HIP"
  git submodule init
  git submodule update
}

build() {
    mkdir -p ${srcdir}/hipSYCL/build
    cd "${srcdir}/hipSYCL/build"
    # We compile with nvcc since Arch Linux uses very new versions of CUDA
    # which may not yet be supported by clang.
    CXX=clang++ cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/opt/hipSYCL/CUDA \
          -DWITH_CPU_BACKEND=ON -DWITH_CUDA_BACKEND=ON -DCMAKE_CXX_FLAGS="--cuda-path=/opt/cuda-9.2 -L/opt/cuda-9.2/lib64" ..
}

package() {
    cd "${srcdir}/hipSYCL/build"
    CXX=clang++ VERBOSE=1 DESTDIR=${pkgdir} make install
    
    echo "
hipSYCL now uses clang as CUDA compiler by default, compilation with nvcc
may or may not work. Since clang 8 does not yet support CUDA 10.1 from Arch Linux,
make sure to point it to an older CUDA version (up to 10.0) when compiling hipSYCL applications.
E.g.: syclcc --cuda-path=/opt/cuda-9.2 -L/opt/cuda-9.2/lib64 ...
    "
}
