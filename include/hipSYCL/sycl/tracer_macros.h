#include <vector>

#ifndef TRACER_MACROS_H
#define TRACER_MACROS_H

#define ALL_TYPES(MACRO)                                                                           \
  MACRO(submit_start, tracer_function_t);                                                          \
  MACRO(submit_end, tracer_function_submit_t);                                                     \
  MACRO(submit_secondary_start, tracer_function_t);                                                \
  MACRO(submit_secondary_end, tracer_function_submit_t);                                           \
  MACRO(parallel_for_start, tracer_function_t);                                                    \
  MACRO(parallel_for_end, tracer_function_t);                                                      \
  MACRO(parallel_for_work_group_start, tracer_function_t);                                         \
  MACRO(parallel_for_work_group_end, tracer_function_t);                                           \
  MACRO(single_task_start, tracer_function_t)                                                      \
  MACRO(single_task_end, tracer_function_t);                                                       \
  MACRO(memcpy_start, tracer_function_t);                                                          \
  MACRO(memcpy_end, tracer_function_t);                                                            \
  MACRO(wait_start, tracer_function_t);                                                            \
  MACRO(wait_end, tracer_function_t);                                                              \
  MACRO(memset_start, tracer_function_t);                                                          \
  MACRO(memset_end, tracer_function_t);                                                            \
  MACRO(fill_start, tracer_function_t);                                                            \
  MACRO(fill_end, tracer_function_t);                                                              \
  MACRO(copy_start, tracer_function_t);                                                            \
  MACRO(copy_end, tracer_function_t);                                                              \
  MACRO(malloc_device_start, tracer_function_t);                                                   \
  MACRO(malloc_device_end, malloc_function_t);                                                     \
  MACRO(malloc_shared_start, tracer_function_t);                                                   \
  MACRO(malloc_shared_end, malloc_function_t);                                                     \
  MACRO(malloc_host_start, tracer_function_t);                                                     \
  MACRO(malloc_host_end, malloc_function_t);                                                       \
  MACRO(free_start, tracer_function_t);                                                            \
  MACRO(free_end, malloc_function_t);                                                              \
  MACRO(finalize, finalizer_function_t);                                                           \
  MACRO(states, void *);

#define MEMBER_VECTOR(name, type) std::vector<type> name;

#define TRACER_FUNCTION1ARG(type)                                                                  \
  for (int i = 0; i < Tracer_utils::tracer_state.size; i++) {                                      \
    if (Tracer_utils::tracer_state.type[i] != nullptr)                                             \
      Tracer_utils::tracer_state.type[i](Tracer_utils::tracer_state.states[i]);                    \
  }

#define TRACER_FUNCTION2ARG(type, arg2)                                                            \
  for (int i = 0; i < Tracer_utils::tracer_state.size; i++) {                                      \
    if (Tracer_utils::tracer_state.type[i] != nullptr)                                             \
      Tracer_utils::tracer_state.type[i](Tracer_utils::tracer_state.states[i], arg2);              \
  }

#define TRACER_FUNCTION3ARG(type, arg2, arg3)                                                      \
  for (int i = 0; i < Tracer_utils::tracer_state.size; i++) {                                      \
    if (Tracer_utils::tracer_state.type[i] != nullptr)                                             \
      Tracer_utils::tracer_state.type[i](Tracer_utils::tracer_state.states[i], arg2, arg3);        \
  }

#endif
