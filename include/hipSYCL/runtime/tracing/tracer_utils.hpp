// #pragma once

#include "tracer_macros.h"
#include <stddef.h>

#ifndef ACPP_TRACER_UTILS_H
#define ACPP_TRACER_UTILS_H

#ifdef __cplusplus
extern "C" {
#endif



typedef void (*tracer_function_t)(void *state);
typedef void (*malloc_function_t)(void *state, void *ptr);
typedef void (*tracer_function_submit_t)(void *state, std::size_t event_hash,
                                         size_t group_node_id);
typedef void (*tracer_function_wait_t)(void *state, size_t event);
typedef void (*tracer_function_depends_on_t)(void *state, size_t event);
typedef void (*tracer_function_true_object_t)(void *state, size_t object_id);
typedef void (*tracer_function_queue_impl_t)(void *state, size_t queue_hash,
                                             bool is_in_order);

typedef void (*finalizer_function_t)(void *);

ALL_TYPES(INIT_FUNCTIONS);

#ifdef __cplusplus
}
#endif

#endif // TRACER_UTILS_H
