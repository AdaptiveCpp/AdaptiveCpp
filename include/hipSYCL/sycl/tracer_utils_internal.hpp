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

  tracer_funcs();
  ~tracer_funcs();

  void set_tracer_equal_num();

  std::size_t size = 0;
  std::vector<tracer_function_t> submit_start;
  std::vector<tracer_function_submit_t> submit_end;
  std::vector<tracer_function_t> submit_secondary_start;
  std::vector<tracer_function_submit_t> submit_secondary_end;
  std::vector<tracer_function_t> parallel_for_start;
  std::vector<tracer_function_t> parallel_for_end;
  std::vector<tracer_function_t> parallel_for_work_group_start;
  std::vector<tracer_function_t> parallel_for_work_group_end;
  std::vector<tracer_function_t> single_task_start;
  std::vector<tracer_function_t> single_task_end;
  std::vector<tracer_function_t> memcpy_start;
  std::vector<tracer_function_t> memcpy_end;
  std::vector<tracer_function_t> wait_start;
  std::vector<tracer_function_t> wait_end;
  std::vector<tracer_function_t> memset_start;
  std::vector<tracer_function_t> memset_end;
  std::vector<tracer_function_t> fill_start;
  std::vector<tracer_function_t> fill_end;
  std::vector<tracer_function_t> copy_start;
  std::vector<tracer_function_t> copy_end;
  std::vector<tracer_function_t> malloc_device_start;
  std::vector<malloc_function_t> malloc_device_end;
  std::vector<tracer_function_t> malloc_host_start;
  std::vector<malloc_function_t> malloc_host_end;
  std::vector<tracer_function_t> malloc_shared_start;
  std::vector<malloc_function_t> malloc_shared_end;
  std::vector<tracer_function_t> free_start;
  std::vector<malloc_function_t> free_end;
  std::vector<finalizer_function_t> finalize;
  std::vector<void *> states;
};

typedef void (*tracer_functs_initialize_t)();

extern tracer_funcs tracer_state;

void initialize_tracers_from_env();

void set_tracer_equal_num(tracer_funcs &);

void finalize_tracing();

extern void (*tracer_func)(tracer_type, tracer_start_end);

void tracer_function(char *function_name, tracer_start_end state);

}; // namespace Tracer_utils

#endif // TRACER_UTILS_H
