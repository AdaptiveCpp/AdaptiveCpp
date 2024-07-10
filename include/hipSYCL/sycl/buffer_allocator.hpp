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
#ifndef HIPSYCL_BUFFER_ALLOCATOR_HPP
#define HIPSYCL_BUFFER_ALLOCATOR_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <limits>
#include <utility>

#include "libkernel/backend.hpp"
#include "exception.hpp"

namespace hipsycl {
namespace sycl {


template <typename T>
using buffer_allocator = std::allocator<T>;

}
}


#endif 
