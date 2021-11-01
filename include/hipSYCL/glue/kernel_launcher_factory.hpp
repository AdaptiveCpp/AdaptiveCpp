/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_KERNEL_LAUNCHER_FACTORY_HPP
#define HIPSYCL_KERNEL_LAUNCHER_FACTORY_HPP

#include <vector>
#include <memory>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/glue/kernel_names.hpp"

#if defined(__HIPSYCL_ENABLE_HIP_TARGET__)
#include "hip/hip_kernel_launcher.hpp"
#endif

#if defined(__HIPSYCL_ENABLE_CUDA_TARGET__)
#include "cuda/cuda_kernel_launcher.hpp"
#endif

#if defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__)
#include "omp/omp_kernel_launcher.hpp"
#endif

#if defined(__HIPSYCL_ENABLE_SPIRV_TARGET__)
#include "ze/ze_kernel_launcher.hpp"
#endif

namespace hipsycl {
namespace glue {

/// Construct kernel launchers.
/// Note: For basic parallel for kernels, local range may argument may be ignored.
///       If it is non-0, it *may* be used as a hint for the backend.
template <class KernelNameTag, rt::kernel_type Type, int Dim, class Kernel,
          typename... Reductions>
std::vector<std::unique_ptr<rt::backend_kernel_launcher>>
make_kernel_launchers(sycl::id<Dim> offset, sycl::range<Dim> local_range,
                      sycl::range<Dim> global_range,
                      std::size_t dynamic_local_memory, Kernel k,
                      Reductions... reductions) {

  using name_traits = kernel_name_traits<KernelNameTag, Kernel>;
  
  std::vector<std::unique_ptr<rt::backend_kernel_launcher>> launchers;
#ifdef __HIPSYCL_ENABLE_HIP_TARGET__
  {
    auto launcher = std::make_unique<hip_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k, reductions...);
    launchers.emplace_back(std::move(launcher));
  }
#endif

#ifdef __HIPSYCL_ENABLE_CUDA_TARGET__
  {
    auto launcher = std::make_unique<cuda_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k, reductions...);
    launchers.emplace_back(std::move(launcher));
  }
#endif

#ifdef __HIPSYCL_ENABLE_SPIRV_TARGET__
  {
    auto launcher = std::make_unique<ze_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k, reductions...);
    launchers.emplace_back(std::move(launcher));
  }
#endif

  // Don't try to compile host kernel during device passes
#if defined(__HIPSYCL_ENABLE_OMPHOST_TARGET__) && \
   !defined(HIPSYCL_LIBKERNEL_DEVICE_PASS)
  {
    auto launcher = std::make_unique<omp_kernel_launcher>();
    launcher->bind<name_traits, Type>(offset, global_range, local_range,
                                      dynamic_local_memory, k, reductions...);
    launchers.emplace_back(std::move(launcher));
  }
#endif
  return launchers;
}
}
}

#endif
