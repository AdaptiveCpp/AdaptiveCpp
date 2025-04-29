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
#ifndef HIPSYCL_CUDA_KERNEL_LAUNCHER_HPP
#define HIPSYCL_CUDA_KERNEL_LAUNCHER_HPP

#include "../generic/hiplike/hiplike_kernel_launcher.hpp"
#include "hipSYCL/runtime/cuda/cuda_queue.hpp"
#include "hipSYCL/runtime/device_id.hpp"

namespace hipsycl {
namespace glue {

using cuda_kernel_launcher =
    hiplike_kernel_launcher<rt::backend_id::cuda, rt::cuda_queue>;

}
}

#endif