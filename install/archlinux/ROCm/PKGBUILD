# Maintainer: Aksel Alpay <alpay at stud dot uni-heidelberg dot de>
pkgname=hipsycl-rocm-git
pkgver=0.6.8
pkgrel=1
pkgdesc="SYCL implementation over CUDA/HIP for AMD devices."
arch=("x86_64")
url="https://github.com/illuhad/hipSYCL"
makedepends=("cmake")
provides=("hipSYCL" "SYCL")
license=("BSD")

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
    export HIPSYCL_PLATFORM=rocm
    mkdir -p ${srcdir}/hipSYCL/build
    cd "${srcdir}/hipSYCL/build"
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/opt/hipSYCL/ROCm ..
}

package() {
    export HIPSYCL_PLATFORM=rocm
    cd "${srcdir}/hipSYCL/build"
    VERBOSE=1 DESTDIR=${pkgdir} make install
}
