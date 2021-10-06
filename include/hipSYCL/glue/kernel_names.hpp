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

#include <utility>
struct __hipsycl_unnamed_kernel {};

namespace hipsycl {
namespace glue {

// This can be used to turn incomplete types of kernel names into complete
// types for backends which e.g. use typeid() to access the mangled name.
template<class KernelName> struct complete_kernel_name {};

template<class KernelName, typename... MultiversionParamaters>
struct multiversioned_kernel_name {};

template<class KernelBodyT, typename... MultiversionParameters>
struct multiversioned_kernel_wrapper {
  template<typename... Args>
  void operator()(Args&&... args) const noexcept {
    kernel(std::forward<Args>(args)...);
  }

  KernelBodyT kernel;
};

template<class KernelNameTag, class KernelBodyT>
struct kernel_name_traits {
  using tag = KernelNameTag;
  // The name that the kernel should have. __hipsycl_unnamed_kernel if
  // unnamed, a type based on the name tag if named.
  using name = complete_kernel_name<tag>;
  // The name that is suggested to be used for name mangling. If unnamed,
  // the kernel body type. Otherwise a type based on the tag.
  using suggested_mangling_name = name;

  template <typename... MultiversionParameters>
  using multiversioned_name =
      multiversioned_kernel_name<name, MultiversionParameters...>;

  template <typename... MultiversionParameters>
  using multiversioned_suggested_mangling_name =
      multiversioned_kernel_name<suggested_mangling_name, MultiversionParameters...>;

  template <typename... MultiversionParameters>
  static auto
  make_multiversioned_kernel_body(const KernelBodyT &body) noexcept {
    return multiversioned_kernel_wrapper<KernelBodyT,
                                         MultiversionParameters...>{body};
  }

  static constexpr bool is_unnamed = false;
};

template<class KernelBodyT>
struct kernel_name_traits<__hipsycl_unnamed_kernel, KernelBodyT> {
  using tag = __hipsycl_unnamed_kernel;
  using name = __hipsycl_unnamed_kernel;
  using suggested_mangling_name = KernelBodyT;

  template <typename... MultiversionParameters>
  using multiversioned_name = __hipsycl_unnamed_kernel;

  template <typename... MultiversionParameters>
  using multiversioned_suggested_mangling_name =
      multiversioned_kernel_wrapper<KernelBodyT, MultiversionParameters...>;

  template <typename... MultiversionParameters>
  static auto
  make_multiversioned_kernel_body(const KernelBodyT &body) noexcept {
    return multiversioned_kernel_wrapper<KernelBodyT,
                                         MultiversionParameters...>{body};
  }

  static constexpr bool is_unnamed = true;
};

}
} // namespace hipsycl

#endif