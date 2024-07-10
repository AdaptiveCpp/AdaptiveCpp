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
#ifndef HIPSYCL_REINTERPRET_POINTER_CAST_HPP
#define HIPSYCL_REINTERPRET_POINTER_CAST_HPP

#include <memory>

namespace hipsycl {
namespace common {
namespace shim {

#ifdef _LIBCPP_VERSION
// libc++ has std::reinterpret_pointer_cast since version 11000
#if _LIBCPP_VERSION < 11000
template< class T, class U > 
std::shared_ptr<T> reinterpret_pointer_cast(const std::shared_ptr<U>& r) noexcept {
    auto p = reinterpret_cast<typename std::shared_ptr<T>::element_type*>(r.get());
    return std::shared_ptr<T>(r, p);
}
#else
using std::reinterpret_pointer_cast;
#endif
#else
// libstdc++ has std::reinterpret_pointer_cast in c++17 mode
using std::reinterpret_pointer_cast;
#endif

}
}
}

#endif

