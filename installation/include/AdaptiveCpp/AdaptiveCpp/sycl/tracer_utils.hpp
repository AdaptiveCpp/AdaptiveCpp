// #pragma once

#include <chrono>
#include <cstdlib>
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
  FINALIZE = 10,
  MALLOC_DEVICE = 11,
  MALLOC_HOST = 12,
  MALLOC_SHARED = 13
};

void initialize_tracer(void (*func)(tracer_start_end), tracer_type, void *);

typedef void (*tracer_function_t)(void *state);
typedef void (*malloc_function_t)(void *state, void *ptr);
typedef void (*tracer_function_submit_t)(void *state, void *event_ptr, void *qptr);
typedef void (*finalizer_function_t)(void *);

void init_state(void *usr_state);

void init_submit_start(tracer_function_t);
void init_submit_secondary_start(tracer_function_t);
void init_parallel_for_start(tracer_function_t);
void init_parallel_for_work_group_start(tracer_function_t);
void init_single_task_start(tracer_function_t);
void init_memcpy_start(tracer_function_t);
void init_wait_start(tracer_function_t);
void init_memset_start(tracer_function_t);
void init_fill_start(tracer_function_t);
void init_copy_start(tracer_function_t);
void init_malloc_device_start(tracer_function_t);
void init_malloc_host_start(tracer_function_t);
void init_malloc_shared_start(tracer_function_t);
void init_free_start(tracer_function_t);

void init_submit_end(tracer_function_submit_t);
void init_submit_secondary_end(tracer_function_submit_t);
void init_parallel_for_end(tracer_function_t);
void init_parallel_for_work_group_end(tracer_function_t);
void init_single_task_end(tracer_function_t);
void init_memcpy_end(tracer_function_t);
void init_wait_end(tracer_function_t);
void init_memset_end(tracer_function_t);
void init_fill_end(tracer_function_t);
void init_copy_end(tracer_function_t);
void init_malloc_device_end(malloc_function_t);
void init_malloc_host_end(malloc_function_t);
void init_malloc_shared_end(malloc_function_t);
void init_free_end(malloc_function_t);

#ifdef __cplusplus
}
#endif

#endif // TRACER_UTILS_H
