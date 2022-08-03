/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2022 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP
#define HIPSYCL_LLVM_SSCP_KERNEL_LAUNCHER_HPP

#include "hipSYCL/glue/generic/code_object.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"

// These variables need to be initialized by the clang plugin.
static std::size_t __hipsycl_local_sscp_hcf_object_id;
static std::size_t __hipsycl_local_sscp_hcf_object_size;
static const char* __hipsycl_local_sscp_hcf_content;

// TODO: Maybe this can be unified with the HIPSYCL_STATIC_HCF_REGISTRATION
// macro. We cannot use this macro directly because it expects
// the object id to be constexpr, which it is not for the SSCP case.
struct __hipsycl_static_sscp_hcf_registration {
  __hipsycl_static_sscp_hcf_registration() {
    ::hipsycl::rt::kernel_cache::get().register_hcf_object(
        ::hipsycl::common::hcf_container{std::string{
            reinterpret_cast<const char *>(__hipsycl_local_sscp_hcf_content),
            __hipsycl_local_sscp_hcf_object_size}});
  }
};
static __hipsycl_static_sscp_hcf_registration
    __hipsycl_register_sscp_hcf_object;

#define __sycl_kernel [[clang::annotate("hipsycl_sscp_kernel")]]

namespace hipsycl {
namespace glue {
namespace sscp_dispatch {

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_single_task(const KernelType &kernel) {
  kernel();
}

template <typename KernelName, typename KernelType>
__sycl_kernel void
kernel_parallel_for(const KernelType& kernel) {
  kernel();
}

}


}
}

#undef __sycl_kernel

#endif
