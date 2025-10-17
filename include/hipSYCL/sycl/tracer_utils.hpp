// #pragma once

#include "tracer_macros.h"
#include <chrono>
#include <dlfcn.h>
#include <vector>

#ifndef TRACER_UTILS_H
#define TRACER_UTILS_H

#define INIT_FUNCTIONS(type, arg_type) void init_##type(arg_type);

template <typename T> struct TD;

#ifdef __cplusplus
extern "C" {
#endif

using hashtype = decltype(std::hash<void *>{}(0));

enum tracer_start_end { START = 0, END = 1 };

typedef void (*tracer_function_t)(void *state);
typedef void (*malloc_function_t)(void *state, void *ptr);
typedef void (*tracer_function_submit_t)(void *state, std::size_t event_hash,
                                         hashtype group_node_id, bool in_order);
typedef void (*tracer_function_wait_t)(void *state, hashtype event);
typedef void (*tracer_function_depends_on_t)(void *state, hashtype event);
typedef void (*tracer_function_true_object_t)(void *state, hashtype event);

typedef void (*finalizer_function_t)(void *);

ALL_TYPES(INIT_FUNCTIONS);

#ifdef __cplusplus
}
#endif

#endif // TRACER_UTILS_H
