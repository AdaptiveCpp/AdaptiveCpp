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
#ifndef HIPSYCL_BIT_CAST_DETAIL_HPP
#define HIPSYCL_BIT_CAST_DETAIL_HPP

#if __has_builtin(__builtin_bit_cast)

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  out = __builtin_bit_cast(Tout, in)

#else

#define HIPSYCL_INPLACE_BIT_CAST(Tin, Tout, in, out)                           \
  {                                                                            \
    union {                                                                    \
      Tout union_out;                                                          \
      Tin union_in;                                                            \
    } u;                                                                       \
    u.union_in = in;                                                           \
    out = u.union_out;                                                         \
  }
#endif



#endif
