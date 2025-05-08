#pragma once

#include <chrono>
#include <dlfcn.h>
#include <unordered_map>

namespace Tracer_utils {

using time_point = std::chrono::high_resolution_clock::time_point;

enum start_end { START = 0, END = 1 };

enum tracer_type {
  SUBMIT = 0,
  SUBMIT_SECONDARY = 1,
  PARALLEL_FOR = 2,
  PARALLEL_FOR_WORK_GROUP = 3,
  SINGLE_TASK = 4,
  MEMCPY = 5,
  WAIT = 6,
  MEMSET = 7,
};

typedef void (*tracer_func_t)(tracer_type, start_end);

extern int size;
extern tracer_func_t *tracer_funcs_array;

void initialize_tracers_from_env();

// extern std::unordered_map<tracer_type, std::string> tracer_type_map;
//
// extern std::unordered_map<tracer_type, std::size_t> Tracer_map;
//
// extern std::unordered_map<tracer_type, double> Tracer_time;

extern void (*tracer_func)(tracer_type, start_end);

void initialize_tracer(void (*func)(tracer_type, start_end));

void tracer_function(char *function_name, start_end state);

}; // namespace Tracer_utils
