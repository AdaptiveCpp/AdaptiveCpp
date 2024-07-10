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
#ifndef HIPSYCL_GLUE_KERNEL_NAMES_HPP
#define HIPSYCL_GLUE_KERNEL_NAMES_HPP

#include <utility>
struct __acpp_unnamed_kernel {};

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
  // The name that the kernel should have. __acpp_unnamed_kernel if
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
struct kernel_name_traits<__acpp_unnamed_kernel, KernelBodyT> {
  using tag = __acpp_unnamed_kernel;
  using name = __acpp_unnamed_kernel;
  using suggested_mangling_name = KernelBodyT;

  template <typename... MultiversionParameters>
  using multiversioned_name = __acpp_unnamed_kernel;

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
} // namespace glue
} // namespace hipsycl

#endif
