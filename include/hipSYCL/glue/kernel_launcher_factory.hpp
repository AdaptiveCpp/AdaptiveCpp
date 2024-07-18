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
#ifndef HIPSYCL_KERNEL_LAUNCHER_FACTORY_HPP
#define HIPSYCL_KERNEL_LAUNCHER_FACTORY_HPP

#include <vector>
#include <memory>

#include "hipSYCL/glue/kernel_launcher_data.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/glue/kernel_names.hpp"
#include "hipSYCL/common/small_vector.hpp"

#if defined(__ACPP_ENABLE_HIP_TARGET__)
#include "hip/hip_kernel_launcher.hpp"
#endif

#if defined(__ACPP_ENABLE_CUDA_TARGET__)
#include "cuda/cuda_kernel_launcher.hpp"
#endif

#if defined(__ACPP_ENABLE_OMPHOST_TARGET__)
#include "omp/omp_kernel_launcher.hpp"
#endif

#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__)
#include "llvm-sscp/sscp_kernel_launcher.hpp"
#endif

namespace hipsycl {
namespace glue {

/// Construct kernel launchers.
/// Note: For basic parallel for kernels, local range may argument may be ignored.
///       If it is non-0, it *may* be used as a hint for the backend.
template <class KernelNameTag, rt::kernel_type Type, int Dim, class Kernel,
          typename... Reductions>
rt::kernel_launcher
make_kernel_launcher(sycl::id<Dim> offset, sycl::range<Dim> local_range,
                     sycl::range<Dim> global_range,
                     std::size_t dynamic_local_memory, Kernel k) {

  using name_traits = kernel_name_traits<KernelNameTag, Kernel>;

  kernel_launcher_data static_launcher_data;
  common::auto_small_vector<std::unique_ptr<rt::backend_kernel_launcher>>
      launchers;
#ifdef __ACPP_ENABLE_HIP_TARGET__
  {
    auto launcher = std::make_unique<hip_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k);
    launchers.emplace_back(std::move(launcher));
  }
#endif

#ifdef __ACPP_ENABLE_CUDA_TARGET__
  {
    auto launcher = std::make_unique<cuda_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k);
    launchers.emplace_back(std::move(launcher));
  }
#endif

#if defined(__ACPP_ENABLE_LLVM_SSCP_TARGET__) && \
  !defined(SYCL_DEVICE_ONLY)
  {
    sscp_kernel_launcher::create<name_traits, Type>(
        static_launcher_data, offset, global_range, local_range,
        dynamic_local_memory, k);
  }
#endif

  // Don't try to compile host kernel during device passes
#if defined(__ACPP_ENABLE_OMPHOST_TARGET__) && \
   !defined(SYCL_DEVICE_ONLY)
  {
    auto launcher = std::make_unique<omp_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k);
    launchers.emplace_back(std::move(launcher));
  }
#endif
  return rt::kernel_launcher{static_launcher_data, std::move(launchers)};
}
}
}

#endif
