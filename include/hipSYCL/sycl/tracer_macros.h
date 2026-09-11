#include <vector>

#ifndef TRACER_MACROS_H
#define TRACER_MACROS_H

#define ALL_TYPES(MACRO)                                                                           \
  MACRO(queue_impl_constructor, tracer_function_queue_impl_t);                                     \
  MACRO(queue_impl_destructor, tracer_function_true_object_t);                                     \
  MACRO(dag_node_constructor, tracer_function_true_object_t);                                      \
  MACRO(dag_node_destructor, tracer_function_true_object_t);                                       \
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
  MACRO(wait_queue_start, tracer_function_t);                                                      \
  MACRO(wait_event_start, tracer_function_t);                                                      \
  MACRO(wait_queue_end, tracer_function_wait_t);                                                   \
  MACRO(wait_event_end, tracer_function_wait_t);                                                   \
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
  MACRO(depends_on_start, tracer_function_t)                                                       \
  MACRO(depends_on_end, tracer_function_depends_on_t)                                              \
  MACRO(finalize, finalizer_function_t);                                                           \
  MACRO(states, void *);

#define MEMBER_VECTOR(name, type) std::vector<type> name;

#define TRACER_FUNCTION_VA_ARGS(type, ...)                                                         \
  for (int i = 0; i < tracer_utils::tracer_state.size; i++) {                                      \
    if (tracer_utils::tracer_state.type[i] != nullptr)                                             \
      tracer_utils::tracer_state.type[i](tracer_utils::tracer_state.states[i], ##__VA_ARGS__);     \
  }

#define TRACER_FUNCTION_VA_ARGS_END(type, ...)                                                     \
  for (int i = tracer_utils::tracer_state.size - 1; i >= 0; i--) {                                 \
    if (tracer_utils::tracer_state.type[i] != nullptr)                                             \
      tracer_utils::tracer_state.type[i](tracer_utils::tracer_state.states[i], ##__VA_ARGS__);     \
  }

#define TRACER_FUNCTION1ARG(type) TRACER_FUNCTION_VA_ARGS(type)
#define TRACER_FUNCTION2ARG(type, arg2) TRACER_FUNCTION_VA_ARGS(type, arg2)
#define TRACER_FUNCTION3ARG(type, arg2, arg3) TRACER_FUNCTION_VA_ARGS(type, arg2, arg3)

#define TRACER_FUNCTION1ARG_END(type) TRACER_FUNCTION_VA_ARGS_END(type)
#define TRACER_FUNCTION2ARG_END(type, arg2) TRACER_FUNCTION_VA_ARGS_END(type, arg2)
#define TRACER_FUNCTION3ARG_END(type, arg2, arg3) TRACER_FUNCTION_VA_ARGS_END(type, arg2, arg3)

#ifndef _WIN32
#define MYLIB_API
#else
#ifdef MYLIB_EXPORTS
#define MYLIB_API __declspec(dllexport)
#else
#define MYLIB_API __declspec(dllimport)
#endif
#endif

#endif // TRACER_MACROS_H
