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

template <typename R, typename First_Arg, typename... Args>
struct function_traits<R (*)(First_Arg, Args...)> {
  using Return_t = R;
  using Args_t = std::tuple<First_Arg, Args...>;
  using Args_tail_t = std::tuple<Args...>;
};

template <typename F> using Return_t = typename function_traits<F>::Return_t;

template <typename F> using Args_t = typename function_traits<F>::Args_t;

template <typename F>
using Args_tail_t = typename function_traits<F>::Args_tail_t;

using time_point = std::chrono::high_resolution_clock::time_point;

#define CALL_FUNCTIONS(type, function_type)                                    \
  Return_t<function_type> call_##type(Args_tail_t<function_type> args);

struct ACPP_COMMON_EXPORT tracer_funcs {

  void initialize_tracer();
  void run_finalizers();
  void set_tracer_equal_num();
  void clear_all();

  std::size_t size = 0;
  ALL_TYPES_NOSTATE(CALL_FUNCTIONS);
  ALL_TYPES(MEMBER_VECTOR);
};

typedef void (*tracer_functs_initialize_t)();

ACPP_COMMON_EXPORT void initialize_tracers_from_env();

ACPP_COMMON_EXPORT void set_tracer_equal_num(tracer_funcs &);

ACPP_COMMON_EXPORT void finalize_tracing();

MYLIB_API extern tracer_funcs tracer_state;
}; // namespace tracer_utils

#endif // TRACER_UTILS_H
