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
#ifndef HIPSYCL_KERNEL_TYPE_HPP
#define HIPSYCL_KERNEL_TYPE_HPP


namespace hipsycl {
namespace rt {


enum class kernel_type {
  single_task,
  basic_parallel_for,
  ndrange_parallel_for,
  hierarchical_parallel_for,
  scoped_parallel_for,
  custom
};



} // namespace rt
} // namespace hipsycl

#endif
