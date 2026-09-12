// #pragma once

#include "hipSYCL/common/export.hpp"
#include "tracer_utils.hpp"
#include <chrono>
#include <tuple>
#include <vector>

#ifndef TRACER_UTILS_INTERNAL_H
#define TRACER_UTILS_INTERNAL_H

namespace tracer_utils {

template <typename F> struct function_traits;

template <typename R, typename... Args> struct function_traits<R (*)(Args...)> {
  using Return_t = R;
  using Args_t = std::tuple<Args...>;
};

template <typename F> using Return_t = typename function_traits<F>::Return_t;

template <typename F> using Args_t = typename function_traits<F>::Args_t;

using time_point = std::chrono::high_resolution_clock::time_point;

#define CALL_FUNCTIONS(type, function_type) void call_##type() {
} // namespace tracer_utils

struct ACPP_COMMON_EXPORT tracer_funcs {

  void initialize_tracer();
  void run_finalizers();
  void set_tracer_equal_num();
  void clear_all();

  template <auto F> void call_tracer(auto... Args) { F(Args...); }

  std::size_t size = 0;
  ALL_TYPES(MEMBER_VECTOR);
};

typedef void (*tracer_functs_initialize_t)();

ACPP_COMMON_EXPORT void initialize_tracers_from_env();

ACPP_COMMON_EXPORT void set_tracer_equal_num(tracer_funcs &);

ACPP_COMMON_EXPORT void finalize_tracing();

ACPP_COMMON_EXPORT extern tracer_funcs tracer_state;
}
; // namespace tracer_utils

#endif // TRACER_UTILS_H
