// #pragma once

#include "hipSYCL/common/export.hpp"
#include <chrono>
#include <vector>

#ifndef TRACER_UTILS_INTERNAL_H
#define TRACER_UTILS_INTERNAL_H

#define MYLIB_EXPORTS
#include "tracer_utils.hpp"

namespace Tracer_utils {

using time_point = std::chrono::high_resolution_clock::time_point;

struct ACPP_COMMON_EXPORT tracer_funcs {

  void initialize_tracer();
  void run_finalizers();
  void set_tracer_equal_num();
  void clear_all();

  std::size_t size = 0;
  ALL_TYPES(MEMBER_VECTOR);
};

typedef void (*tracer_functs_initialize_t)();

ACPP_COMMON_EXPORT void initialize_tracers_from_env();

ACPP_COMMON_EXPORT void set_tracer_equal_num(tracer_funcs &);

ACPP_COMMON_EXPORT void finalize_tracing();

ACPP_COMMON_EXPORT void tracer_function(char *function_name, tracer_start_end state);

MYLIB_API extern tracer_funcs tracer_state;

}; // namespace Tracer_utils

#endif // TRACER_UTILS_H
