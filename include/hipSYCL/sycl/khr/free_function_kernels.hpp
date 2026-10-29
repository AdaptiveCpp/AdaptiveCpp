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
#include "hipSYCL/sycl/libkernel/nd_range.hpp"
#include "hipSYCL/sycl/libkernel/nd_item.hpp"

#include <array>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

namespace hipsycl {
namespace sycl {
namespace khr {

enum class property_id : std::uint32_t {
  requires_work_group = 1,
  sub_group_size = 2
};

template <unsigned N>
struct requires_work_group_property {
  static constexpr std::uint32_t id =
      static_cast<std::uint32_t>(property_id::requires_work_group);
  static constexpr std::uint32_t value = N;
};

template <unsigned N>
inline constexpr requires_work_group_property<N> requires_work_group{};

template <unsigned N>
struct sub_group_size_property {
  static constexpr std::uint32_t id =
      static_cast<std::uint32_t>(property_id::sub_group_size);
  static constexpr std::uint32_t value = N;
};

template <unsigned N>
inline constexpr sub_group_size_property<N> sub_group_size{};

template <auto *Func>
struct kernel_function_s {};

template <auto *Func>
inline constexpr kernel_function_s<Func> kernel_function{};

namespace detail {

struct property_record {
  std::uint32_t id;
  std::uint32_t value;
};

template <typename... Props>
struct properties_object {
  std::uint32_t count;
  std::array<property_record, sizeof...(Props)> entries;
};

template <>
struct properties_object<> {
  std::uint32_t count;
};

template <typename... Props>
[[gnu::used]]
inline constexpr properties_object<Props...> property_global = {
    sizeof...(Props),
    {{{Props::id, Props::value}...}}
};

template <>
[[gnu::used]]
inline constexpr properties_object<> property_global<> = {0};

template <typename... Props>
constexpr auto property_global_ptr(Props...) {
  return &property_global<Props...>;
}

template <class F>
struct function_parameter_types;

template <class R, class... Params>
struct function_parameter_types<R (*)(Params...)> {
  using tuple_type = std::tuple<std::decay_t<Params>...>;
};

template <class R, class... Params>
struct function_parameter_types<R (*)(Params...) noexcept> {
  using tuple_type = std::tuple<std::decay_t<Params>...>;
};

template <auto *Func, typename... Args>
constexpr void check_launch_constraints() {
  static_assert(std::is_invocable_v<decltype(Func), Args...>,
                "The arguments passed to the free function kernel launch are "
                "not invocable on the kernel function.");
  static_assert(
      (is_device_copyable_v<std::decay_t<Args>> && ...),
      "All free function kernel arguments must be device copyable.");
}

template <auto *Func, class TupleType>
struct free_function_task_wrapper {
  TupleType _args;

  template <std::size_t... Is>
  void call_impl(std::index_sequence<Is...>) const {
    Func(std::get<Is>(_args)...);
  }

  void operator()() const {
    call_impl(std::make_index_sequence<std::tuple_size_v<TupleType>>{});
  }
};

template <auto *Func, int Dims, class TupleType>
struct free_function_grouped_wrapper {
  TupleType _args;

  template <std::size_t... Is>
  void call_impl(std::index_sequence<Is...>) const {
    Func(std::get<Is>(_args)...);
  }

  void operator()(sycl::nd_item<Dims>) const {
    call_impl(std::make_index_sequence<std::tuple_size_v<TupleType>>{});
  }
};

template <auto *Func, typename... Args>
auto make_converted_args_tuple(Args &&...args) {
  using tuple_type =
      typename function_parameter_types<decltype(Func)>::tuple_type;
  return tuple_type{std::forward<Args>(args)...};
}

} // namespace detail

// launch_task ---------------------------------------------------------------

template <auto *Func, typename... Args>
void launch_task(handler &h, kernel_function_s<Func>, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  auto packed = detail::make_converted_args_tuple<Func>(std::forward<Args>(args)...);
  h.single_task(
      detail::free_function_task_wrapper<Func, decltype(packed)>{std::move(packed)});
}

template <auto *Func, typename... Args>
void launch_task(queue &q, kernel_function_s<Func> k, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  auto packed = detail::make_converted_args_tuple<Func>(std::forward<Args>(args)...);
  q.submit([=](handler &h) {
    h.single_task(
        detail::free_function_task_wrapper<Func, decltype(packed)>{packed});
  });
}

// launch_grouped ------------------------------------------------------------

template <auto *Func, int Dims, typename... Args>
void launch_grouped(handler &h, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func>, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  auto packed = detail::make_converted_args_tuple<Func>(std::forward<Args>(args)...);
  h.parallel_for(
      nd_range<Dims>{global, local},
      detail::free_function_grouped_wrapper<Func, Dims, decltype(packed)>{
          std::move(packed)});
}

template <auto *Func, int Dims, typename... Args>
void launch_grouped(queue &q, range<Dims> global, range<Dims> local,
                    kernel_function_s<Func> k, Args &&...args) {
  detail::check_launch_constraints<Func, Args...>();
  auto packed = detail::make_converted_args_tuple<Func>(std::forward<Args>(args)...);
  q.submit([=](handler &h) {
    h.parallel_for(
        nd_range<Dims>{global, local},
        detail::free_function_grouped_wrapper<Func, Dims, decltype(packed)>{
            packed});
  });
}

} // namespace khr
} // namespace sycl
} // namespace hipsycl

#if ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP && !defined(__ACPP_USE_ACCELERATED_CPU__)
#define SYCL_KHR_FUNCTION(...)                                                 \
  [[clang::annotate("hipsycl_sscp_outlining")]]                                \
  [[clang::annotate(                                                           \
      "sycl.function.properties",                                              \
      ::hipsycl::sycl::khr::detail::property_global_ptr(__VA_ARGS__))]]
#else
#define SYCL_KHR_FUNCTION(...)
#endif

#define SYCL_KHR_KERNEL(...) SYCL_KHR_FUNCTION(__VA_ARGS__)

#endif
