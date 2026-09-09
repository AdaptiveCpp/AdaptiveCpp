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
#ifndef HIPSYCL_KHR_FREE_FUNCTION_KERNELS_HPP
#define HIPSYCL_KHR_FREE_FUNCTION_KERNELS_HPP

#include "hipSYCL/sycl/handler.hpp"
#include "hipSYCL/sycl/queue.hpp"
#include "hipSYCL/sycl/is_device_copyable.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"

#include <type_traits>
#include <utility>

// Implementation of the sycl_khr_free_function_kernels extension.
//
// A free function kernel is an ordinary namespace-scope C++ function decorated
// with the SYCL_KHR_KERNEL() macro. The function itself becomes the device
// kernel entry point; its parameters are the kernel arguments (each must be
// device-copyable), giving them a defined order. The kernel is identified by
// the function itself and launched via a kernel_function<Func> handle.
//
// In AdaptiveCpp the macro maps onto the existing SSCP kernel-entry-point
// annotations, which cause the generic single-pass compiler to outline the
// function as a kernel and register it in the HCF under its mangled symbol name.
// Note that (unlike the AdaptiveCpp PCUDA free kernels) we do NOT add the
// "acpp_free_kernel" annotation: that would turn an ordinary host call to the
// function into a kernel launch, whereas this extension launches only through
// the explicit launch_* functions below.

// Decorate a namespace-scope function declaration to define a free function
// kernel. The macro takes no arguments.
#define SYCL_KHR_KERNEL()                                                      \
  [[clang::annotate("hipsycl_sscp_kernel")]]                                   \
  [[clang::annotate("hipsycl_sscp_outlining")]]

namespace hipsycl {
namespace sycl {
namespace khr {

// Handle identifying the free function kernel to launch. The kernel is
// identified by the value kernel_function<Func>, not by an explicit template
// argument on the launch function.
template <auto *Func> struct kernel_function_s {};

template <auto *Func>
inline constexpr kernel_function_s<Func> kernel_function{};

namespace detail {

template <auto *Func, typename... Args>
constexpr void check_launch_constraints() {
  static_assert(std::is_invocable_v<decltype(Func), Args...>,
                "The arguments passed to the free function kernel launch are "
                "not invocable on the kernel function.");
  static_assert(
      (is_device_copyable_v<std::decay_t<Args>> && ...),
      "All free function kernel arguments must be device copyable.");
}

} // namespace detail

// launch_task ---------------------------------------------------------------

template <auto *Func, typename... Args>
void launch_task(handler &h, kernel_function_s<Func>, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  h.template AdaptiveCpp_launch_free_function_kernel<Func, 1>(
      range<1>{1}, range<1>{1}, std::forward<Args>(args)...);
}

template <auto *Func, typename... Args>
void launch_task(queue &q, kernel_function_s<Func> k, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  q.submit([&](handler &h) {
    launch_task<Func>(h, k, std::forward<Args>(args)...);
  });
}

// launch_grouped ------------------------------------------------------------

template <auto *Func, int Dims, typename... Args>
void launch_grouped(handler &h, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func>, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  h.template AdaptiveCpp_launch_free_function_kernel<Func, Dims>(
      global, local, std::forward<Args>(args)...);
}

template <auto *Func, int Dims, typename... Args>
void launch_grouped(queue &q, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func> k, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  q.submit([&](handler &h) {
    launch_grouped<Func>(h, global, local, k, std::forward<Args>(args)...);
  });
}

} // namespace khr
} // namespace sycl
} // namespace hipsycl

#endif
