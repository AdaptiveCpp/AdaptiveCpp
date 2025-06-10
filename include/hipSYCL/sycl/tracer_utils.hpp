// #pragma once

#include <chrono>
#include <dlfcn.h>
#include <vector>

#ifndef TRACER_UTILS_H
#define TRACER_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif

enum tracer_start_end { START = 0, END = 1 };

enum tracer_type {
  SUBMIT = 0,
  SUBMIT_SECONDARY = 1,
  PARALLEL_FOR = 2,
  PARALLEL_FOR_WORK_GROUP = 3,
  SINGLE_TASK = 4,
  MEMCPY = 5,
  WAIT = 6,
  MEMSET = 7,
  FILL = 8,
  COPY = 9,
  FINALIZE = 10
};

void initialize_tracer(void (*func)(tracer_start_end), tracer_type, void *);

#ifdef __cplusplus
}
#endif

#endif // TRACER_UTILS_H
