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

  std::size_t size = 0;
  std::vector<tracer_function_t> submit_start;
  std::vector<tracer_function_t> submit_end;
  std::vector<void *> submit_state;
  std::vector<tracer_function_t> submit_secondary_start;
  std::vector<tracer_function_t> submit_secondary_end;
  std::vector<void *> submit_secondary_state;
  std::vector<tracer_function_t> parallel_for_start;
  std::vector<tracer_function_t> parallel_for_end;
  std::vector<void *> parallel_for_state;
  std::vector<tracer_function_t> parallel_for_work_group_start;
  std::vector<tracer_function_t> parallel_for_work_group_end;
  std::vector<void *> parallel_for_work_group_state;
  std::vector<tracer_function_t> single_task_start;
  std::vector<tracer_function_t> single_task_end;
  std::vector<void *> single_task_state;
  std::vector<tracer_function_t> memcpy_start;
  std::vector<tracer_function_t> memcpy_end;
  std::vector<void *> memcpy_state;
  std::vector<tracer_function_t> wait_start;
  std::vector<tracer_function_t> wait_end;
  std::vector<void *> wait_state;
  std::vector<tracer_function_t> memset_start;
  std::vector<tracer_function_t> memset_end;
  std::vector<void *> memset_state;
  std::vector<tracer_function_t> fill_start;
  std::vector<tracer_function_t> fill_end;
  std::vector<void *> fill_state;
  std::vector<tracer_function_t> copy_start;
  std::vector<tracer_function_t> copy_end;
  std::vector<void *> copy_state;
  std::vector<finalizer_function_t> finalize;
};

typedef void (*tracer_functs_initialize_t)(tracer_funcs &);

extern tracer_funcs tracer_state;

void initialize_tracers_from_env();

void set_tracer_equal_num(tracer_funcs &);

void finalize_tracing();

extern void (*tracer_func)(tracer_type, tracer_start_end);

void tracer_function(char *function_name, tracer_start_end state);

}; // namespace Tracer_utils

#endif // TRACER_UTILS_H
