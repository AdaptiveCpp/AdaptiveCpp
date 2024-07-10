/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_BUILTIN_KERNELS_HPP
#define HIPSYCL_BUILTIN_KERNELS_HPP

#include "backend.hpp"

#include "accessor.hpp"
#include "id.hpp"

#include "hipSYCL/sycl/access.hpp"

namespace hipsycl {
namespace sycl {

namespace detail::kernels {

template <class T, int Dim, access::mode Mode, access::target Tgt,
          accessor_variant V>
class fill_kernel {
public:
  fill_kernel(sycl::accessor<T, Dim, Mode, Tgt, V> dest, const T &src)
      : _dest{dest}, _src{src} {}

  void operator()(sycl::id<Dim> tid) const { _dest[tid] = _src; }

private:
  sycl::accessor<T, Dim, Mode, Tgt, V> _dest;
  T _src;
};

template<class T>
class fill_kernel_usm {
public:
  fill_kernel_usm(T* ptr, T value)
      : _dest{ptr}, _src{value} {}

  void operator()(sycl::id<1> tid) const { _dest[tid[0]] = _src; }

private:
  T* _dest;
  T _src;
};

}


}
}

#endif