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
#ifndef ACPP_OMP_PHYS_MEM_HPP
#define ACPP_OMP_PHYS_MEM_HPP

#include <cstddef>
#include <optional>

namespace hipsycl {
namespace rt {

/**
 * @brief Get the amount of physical memory (RAM) available on the system, in
 * bytes.
 *
 * @return The amount of physical memory available, or std::nullopt if the
 * information cannot be retrieved.
 *
 * @details This function is implemented for Mac OS X and Linux. Other platforms
 * will return std::nullopt.
 */
std::optional<std::size_t> get_physical_memory();
} // namespace rt
} // namespace hipsycl

#endif