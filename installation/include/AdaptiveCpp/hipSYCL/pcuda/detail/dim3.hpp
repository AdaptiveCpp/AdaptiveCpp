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


#ifndef ACPP_PCUDA_DIM3_HPP
#define ACPP_PCUDA_DIM3_HPP

struct dim3 {
  dim3(unsigned x_=1, unsigned y_=1, unsigned z_=1)
  : x{x_}, y{y_}, z{z_} {}

  unsigned x;
  unsigned y;
  unsigned z;
};

#endif
