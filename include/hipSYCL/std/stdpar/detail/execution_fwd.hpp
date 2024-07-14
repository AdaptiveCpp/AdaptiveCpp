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

#include <pstl/execution_defs.h>

namespace hipsycl::stdpar {

using par_unseq =
    __pstl::execution::parallel_unsequenced_policy;

using par =
    __pstl::execution::parallel_policy;

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
