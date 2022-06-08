/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_GLUE_CODE_OBJECT_HPP
#define HIPSYCL_GLUE_CODE_OBJECT_HPP


#include <vector>
#include <array>
#include <string>
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/common/hcf_container.hpp"

#define HIPSYCL_STATIC_KERNEL_REGISTRATION(KernelT) \
  (void)::hipsycl::rt::detail::static_kernel_registration<KernelT>::init;

template<std::size_t Hcf_object_id>
struct __hipsycl_hcf_registration {};

#define HIPSYCL_STATIC_HCF_REGISTRATION(hcf_object_id, hcf_string, hcf_size)   \
  template <> struct __hipsycl_hcf_registration<hcf_object_id> {               \
    __hipsycl_hcf_registration() {                                             \
      ::hipsycl::rt::kernel_cache::get().register_hcf_object(                  \
          ::hipsycl::common::hcf_container{std::string{                        \
              reinterpret_cast<const char *>(hcf_string), hcf_size}});         \
    }                                                                          \
  };                                                                           \
  __hipsycl_hcf_registration<hcf_object_id>                                    \
      __hipsycl_register_hcf_##hcf_object_id;

#ifdef __HIPSYCL_MULTIPASS_CUDA_HEADER__
 #include __HIPSYCL_MULTIPASS_CUDA_HEADER__
#endif

#ifdef __HIPSYCL_MULTIPASS_SPIRV_HEADER__
 #include __HIPSYCL_MULTIPASS_SPIRV_HEADER__
#endif

#endif
