# Maintainer: Aksel Alpay <alpay at stud dot uni-heidelberg dot de>
pkgname=hipsycl-cuda-git
pkgver=0.6.8
pkgrel=1
pkgdesc="SYCL implementation over CUDA/HIP for NVIDIA devices."
arch=("x86_64")
url="https://github.com/illuhad/hipSYCL"
makedepends=("cmake")
provides=("hipSYCL" "SYCL")
license=("BSD")
depends=("cuda")

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
    export HIPSYCL_PLATFORM=cuda
    mkdir -p ${srcdir}/hipSYCL/build
    cd "${srcdir}/hipSYCL/build"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/opt/hipSYCL/CUDA ..
}

package() {
    export HIPSYCL_PLATFORM=cuda
    cd "${srcdir}/hipSYCL/build"
    VERBOSE=1 DESTDIR=${pkgdir} make install
}
