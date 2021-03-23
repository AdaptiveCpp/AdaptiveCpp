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

#ifndef HIPSYCL_GLUE_KERNEL_NAMES_HPP
#define HIPSYCL_GLUE_KERNEL_NAMES_HPP

struct __hipsycl_unnamed_kernel {};

namespace hipsycl {
namespace glue {

template<class KernelName> struct complete_kernel_name {};

// This can be used to turn incomplete types of kernel names into complete
// types for backends which e.g. use typeid() to access the mangled name.
template <class Name, class KernelBody> struct kernel_name {
  using complete_type = complete_kernel_name<Name>;
  using effective_type = Name;
};
template <class KernelBody>
struct kernel_name<__hipsycl_unnamed_kernel, KernelBody> {
  using complete_type = __hipsycl_unnamed_kernel;
  using effective_type = KernelBody;
};

template <class Name, class KernelBody>
using complete_kernel_name_t = 
  typename kernel_name<Name, KernelBody>::complete_type;


template <class Name, class KernelBody>
using effective_kernel_name_t = 
  typename kernel_name<Name, KernelBody>::effective_type;
}
} // namespace hipsycl

#endif