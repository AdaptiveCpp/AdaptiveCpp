// #pragma once

#include "tracer_utils.hpp"
#include <chrono>
#include <dlfcn.h>
#include <vector>

#ifndef TRACER_UTILS_INTERNAL_H
#define TRACER_UTILS_INTERNAL_H

namespace Tracer_utils {

using time_point = std::chrono::high_resolution_clock::time_point;

struct tracer_funcs {

  void initialize_tracer();

  void run_finalizers();
  void set_tracer_equal_num();
  void clear_all();

  std::size_t size = 0;
  ALL_TYPES(MEMBER_VECTOR);
};

typedef void (*tracer_functs_initialize_t)();

void initialize_tracers_from_env();

void set_tracer_equal_num(tracer_funcs &);

void finalize_tracing();

extern tracer_funcs tracer_state;

void tracer_function(char *function_name, tracer_start_end state);

}; // namespace Tracer_utils

#endif // TRACER_UTILS_H
