/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */


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