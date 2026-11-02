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
#ifndef HIPSYCL_PSTL_EXECUTION_FWD_HPP
#define HIPSYCL_PSTL_EXECUTION_FWD_HPP

#include <type_traits>

#ifdef _LIBCPP_VERSION
// Load policy definitions without pulling in the AdaptiveCpp implementations
// while the SYCL runtime headers are still being parsed.
#define HIPSYCL_STDPAR_EXECUTION_FWD_ONLY
#include <execution>
#undef HIPSYCL_STDPAR_EXECUTION_FWD_ONLY

#ifdef _LIBCPP_HAS_NO_INCOMPLETE_PSTL
#error "AdaptiveCpp stdpar requires libc++ execution policies; try -fexperimental-library"
#endif

namespace hipsycl::stdpar {
using par_unseq = std::execution::parallel_unsequenced_policy;
using par = std::execution::parallel_policy;
}
#else
// Keep the policy-only include: loading all of <execution> here changes
// qualified lookup in templates declared before the user's <execution> include.
#include <pstl/execution_defs.h>

namespace hipsycl::stdpar {
using par_unseq = __pstl::execution::parallel_unsequenced_policy;
using par = __pstl::execution::parallel_policy;

struct par_unseq_host_fallback_policy
    : public __pstl::execution::parallel_unsequenced_policy {};

struct par_host_fallback_policy
    : public __pstl::execution::parallel_policy {};

inline constexpr par_unseq_host_fallback_policy par_unseq_host_fallback {};
inline constexpr par_host_fallback_policy par_host_fallback {};
}

namespace __pstl::execution {
template <>
struct is_execution_policy<hipsycl::stdpar::par_unseq_host_fallback_policy>
    : std::true_type {};

template <>
struct is_execution_policy<hipsycl::stdpar::par_host_fallback_policy>
    : std::true_type {};
}
#endif

namespace hipsycl::stdpar {
// libc++ policies cannot be copied. Preserve by-value interception for copyable
// policies: otherwise the native ExecutionPolicy&& overload wins for mutable
// lvalues and rvalues in libstdc++.
using par_unseq_argument =
    std::conditional_t<std::is_copy_constructible_v<par_unseq>,
                       par_unseq, const par_unseq&>;
} // namespace hipsycl::stdpar


#endif
