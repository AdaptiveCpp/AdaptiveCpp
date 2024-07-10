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
#ifndef HIPSYCL_MEMORY_HPP
#define HIPSYCL_MEMORY_HPP

// Note: This file defines enumerations that are also used by low-level SSCP
// builtin definitions. Keep this as a file without include dependencies!

namespace hipsycl {
namespace sycl {

// Do not change the order of these enums, as the compiler may generate
// atomic calls based on the int values of these enums
enum class memory_scope : int {
  work_item,
  sub_group,
  work_group,
  device,
  system
};

inline constexpr auto memory_scope_work_item = memory_scope::work_item;
inline constexpr auto memory_scope_sub_group = memory_scope::sub_group;
inline constexpr auto memory_scope_work_group = memory_scope::work_group;
inline constexpr auto memory_scope_device = memory_scope::device;
inline constexpr auto memory_scope_system = memory_scope::system;

// Do not change the order of these enums, as the compiler may generate
// atomic calls based on the int values of these enums
enum class memory_order : int
{
  relaxed,
  acquire,
  release,
  acq_rel,
  seq_cst
};

inline constexpr auto memory_order_relaxed = memory_order::relaxed;
inline constexpr auto memory_order_acquire = memory_order::acquire;
inline constexpr auto memory_order_release = memory_order::release;
inline constexpr auto memory_order_acq_rel = memory_order::acq_rel;
inline constexpr auto memory_order_seq_cst = memory_order::seq_cst;

namespace access {

// Do not change the order of these enums, as the compiler may generate
// atomic calls based on the int values of these enums
enum class address_space : int
{
  global_space,
  local_space,
  constant_space,
  private_space,
  generic_space
};

} // namespace access

}
} // namespace hipsycl

#endif
