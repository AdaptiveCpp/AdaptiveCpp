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
#ifndef HIPSYCL_SYCL_ASPECT_HPP
#define HIPSYCL_SYCL_ASPECT_HPP

#include <type_traits>

namespace hipsycl {
namespace sycl {

enum class aspect {
  cpu,
  gpu,
  accelerator,
  custom,
  emulated,
  host_debuggable,
  fp16,
  fp64,
  atomic64,
  image,
  online_compiler,
  online_linker,
  queue_profiling,
  usm_device_allocations,
  usm_host_allocations,
  usm_atomic_host_allocations,
  usm_shared_allocations,
  usm_atomic_shared_allocations,
  usm_system_allocations
};

template <aspect Aspect> struct any_device_has : public std::true_type {};
template <aspect Aspect> struct all_devices_have : public std::false_type {};

// We always have a CPU device that is host debuggable
template <>
struct any_device_has<aspect::host_debuggable> : public std::true_type {};

// Images are unsupported
template <>
struct any_device_has<aspect::image> : public std::false_type {};

// All backends in hipSYCL must support at least explicit USM by design
template <>
struct all_devices_have<aspect::usm_device_allocations>
    : public std::true_type {};
template <>
struct all_devices_have<aspect::usm_atomic_host_allocations>
    : public std::true_type {};

template <aspect A>
inline constexpr bool any_device_has_v = any_device_has<A>::value;

template <aspect A>
inline constexpr bool all_devices_have_v = all_devices_have<A>::value;

}
}

#endif
