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

#ifndef ACPP_GLUE_CODE_OBJECT_HPP
#define ACPP_GLUE_CODE_OBJECT_HPP



#include <vector>
#include <array>
#include <string>
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/common/hcf_container.hpp"

#define ACPP_STATIC_KERNEL_REGISTRATION(KernelT) \
  (void)::hipsycl::rt::detail::static_kernel_registration<KernelT>::init;

#define ACPP_STATIC_HCF_REGISTRATION(hcf_obj, hcf_string, hcf_size)            \
  class __acpp_hcf_registration##hcf_obj {                                     \
  private:                                                                     \
    ::hipsycl::rt::hcf_object_id _id;                                          \
                                                                               \
  public:                                                                      \
    __acpp_hcf_registration##hcf_obj() {                                       \
      this->_id = ::hipsycl::rt::hcf_cache::get().register_hcf_object(         \
          ::hipsycl::common::hcf_container{std::string{                        \
              reinterpret_cast<const char *>(hcf_string), hcf_size}});         \
    }                                                                          \
    ~__acpp_hcf_registration##hcf_obj() {                                      \
      ::hipsycl::rt::hcf_cache::get().unregister_hcf_object(this->_id);        \
    }                                                                          \
  };                                                                           \
  static __acpp_hcf_registration##hcf_obj __acpp_hcf_obj##hcf_obj;

#ifdef __ACPP_MULTIPASS_CUDA_HEADER__
 #include __ACPP_MULTIPASS_CUDA_HEADER__
#endif

#ifdef __ACPP_MULTIPASS_HIP_HEADER__
 #include __ACPP_MULTIPASS_HIP_HEADER__
#endif

#endif
