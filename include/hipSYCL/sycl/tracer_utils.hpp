// #pragma once

#include <chrono>
#include <dlfcn.h>
#include <vector>

#ifndef TRACER_UTILS_H
#define TRACER_UTILS_H

namespace Tracer_utils {

using time_point = std::chrono::high_resolution_clock::time_point;

enum class start_end { START = 0, END = 1 };

enum class tracer_type {
  SUBMIT = 0,
  SUBMIT_SECONDARY = 1,
  PARALLEL_FOR = 2,
  PARALLEL_FOR_WORK_GROUP = 3,
  SINGLE_TASK = 4,
  MEMCPY = 5,
  WAIT = 6,
  MEMSET = 7,
};

typedef void (*tracer_function_t)(start_end);

struct tracer_funcs {
  std::vector<tracer_function_t> submit;
  std::vector<tracer_function_t> submit_secondary;
  std::vector<tracer_function_t> parallel_for;
  std::vector<tracer_function_t> parallel_for_work_group;
  std::vector<tracer_function_t> single_task;
  std::vector<tracer_function_t> memcpy;
  std::vector<tracer_function_t> wait;
  std::vector<tracer_function_t> memset;
  std::vector<tracer_function_t> fill;
  std::vector<tracer_function_t> copy;
  std::vector<void (*)()> finalize;
};

typedef void (*tracer_functs_initialize_t)(tracer_funcs &);

extern tracer_funcs tracer_state;

void initialize_tracers_from_env();

void finalize_tracing();

extern void (*tracer_func)(tracer_type, start_end);

void initialize_tracer(void (*func)(tracer_type, start_end));

void tracer_function(char *function_name, start_end state);

}; // namespace Tracer_utils

#endif // TRACER_UTILS_H
