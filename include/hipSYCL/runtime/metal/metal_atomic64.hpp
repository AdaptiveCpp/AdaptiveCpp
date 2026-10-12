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
#ifndef HIPSYCL_METAL_ATOMIC64_HPP
#define HIPSYCL_METAL_ATOMIC64_HPP

namespace hipsycl {
namespace rt {

// Shared by the runtime and the Metal kernel library, must be a power of two
inline constexpr unsigned metal_atomic64_lock_table_size = 16384;

} // namespace rt
} // namespace hipsycl

#endif
