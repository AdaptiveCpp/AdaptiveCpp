/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_SCHEDULING_UTIL_HPP
#define HIPSYCL_SCHEDULING_UTIL_HPP

#include <cassert>

#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/range.hpp"

namespace hipsycl {
namespace rt {

template<class U, class T>
bool dynamic_is(T* val)
{
  return dynamic_cast<U*>(val) != nullptr;
}

template<class U, class T>
void assert_is(T* val)
{
  assert(dynamic_is<U>(val));
}

template<class U, class T>
U* cast(T* val)
{
  assert_is<U>(val);
  return static_cast<U*>(val);
}


template<int Dim>
sycl::id<3> embed_in_id3(sycl::id<Dim> idx) {
  static_assert(Dim >= 1 && Dim <=3, 
      "id dim must be between 1 and 3");

  if constexpr(Dim == 1) {
    return sycl::id<3>{0, 0, idx[0]};
  } else if constexpr(Dim == 2) {
    return sycl::id<3>{0,idx[0], idx[1]};
  } else if constexpr(Dim == 3) {
    return idx;
  }
}

template<int Dim>
sycl::range<3> embed_in_range3(sycl::range<Dim> r) {
  static_assert(Dim >= 1 && Dim <=3, 
      "range dim must be between 1 and 3");

  if constexpr(Dim == 1) {
    return sycl::range<3>{1, 1, r[0]};
  } else if constexpr(Dim == 2) {
    return sycl::range<3>{1, r[0], r[1]};
  } else if constexpr(Dim == 3) {
    return r;
  }
}

}
}

#endif
