// #pragma once

#include "tracer_macros.h"
#include <chrono>
#include <dlfcn.h>
#include <vector>

#ifndef TRACER_UTILS_H
#define TRACER_UTILS_H

#define INIT_FUNCTIONS(type, arg_type) void init_##type(arg_type);

#ifdef __cplusplus
extern "C" {
#endif

enum tracer_start_end { START = 0, END = 1 };

typedef void (*tracer_function_t)(void *state);
typedef void (*malloc_function_t)(void *state, void *ptr);
typedef void (*tracer_function_submit_t)(void *state, void *event_ptr, void *qptr);
typedef void (*finalizer_function_t)(void *);

ALL_TYPES(INIT_FUNCTIONS);

#ifdef __cplusplus
}
#endif

#endif // TRACER_UTILS_H
