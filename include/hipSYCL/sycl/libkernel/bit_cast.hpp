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
#ifndef HIPSYCL_BIT_CAST_HPP
#define HIPSYCL_BIT_CAST_HPP

#include "backend.hpp"
#include "detail/bit_cast.hpp"

namespace hipsycl {
namespace sycl {

template <class Tout, class Tin>
ACPP_UNIVERSAL_TARGET
Tout bit_cast(Tin x) {
  Tout result;
  HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, x, result);
  return result;
}


}
}

#endif
