#ifdef __cplusplus
#include <vector>
#endif

#ifndef ACPP_TRACER_MACROS_H
#define ACPP_TRACER_MACROS_H

#define ALL_TYPES_NOSTATE_BEGIN(MACRO)                                         \
  MACRO(queue_impl_constructor, tracer_function_queue_impl_t);                 \
  MACRO(dag_node_constructor, tracer_function_true_object_t);                  \
  MACRO(submit_start, tracer_function_t);                                      \
  MACRO(submit_secondary_start, tracer_function_t);                            \
  MACRO(parallel_for_start, tracer_function_t);                                \
  MACRO(parallel_for_work_group_start, tracer_function_t);                     \
  MACRO(single_task_start, tracer_function_t);                                 \
  MACRO(memcpy_start, tracer_function_t);                                      \
  MACRO(wait_queue_start, tracer_function_t);                                  \
  MACRO(wait_event_start, tracer_function_t);                                  \
  MACRO(memset_start, tracer_function_t);                                      \
  MACRO(fill_start, tracer_function_t);                                        \
  MACRO(copy_start, tracer_function_t);                                        \
  MACRO(malloc_device_start, tracer_function_t);                               \
  MACRO(malloc_shared_start, tracer_function_t);                               \
  MACRO(malloc_host_start, tracer_function_t);                                 \
  MACRO(free_start, tracer_function_t);                                        \
  MACRO(depends_on_start, tracer_function_t);

#define ALL_TYPES_NOSTATE_END(MACRO)                                           \
  MACRO(queue_impl_destructor, tracer_function_true_object_t);                 \
  MACRO(dag_node_destructor, tracer_function_true_object_t);                   \
  MACRO(submit_end, tracer_function_submit_t);                                 \
  MACRO(submit_secondary_end, tracer_function_submit_t);                       \
  MACRO(parallel_for_end, tracer_function_t);                                  \
  MACRO(parallel_for_work_group_end, tracer_function_t);                       \
  MACRO(single_task_end, tracer_function_t);                                   \
  MACRO(memcpy_end, tracer_function_t);                                        \
  MACRO(wait_queue_end, tracer_function_wait_t);                               \
  MACRO(wait_event_end, tracer_function_wait_t);                               \
  MACRO(memset_end, tracer_function_t);                                        \
  MACRO(fill_end, tracer_function_t);                                          \
  MACRO(copy_end, tracer_function_t);                                          \
  MACRO(malloc_device_end, malloc_function_t);                                 \
  MACRO(malloc_shared_end, malloc_function_t);                                 \
  MACRO(malloc_host_end, malloc_function_t);                                   \
  MACRO(free_end, malloc_function_t);                                          \
  MACRO(depends_on_end, tracer_function_depends_on_t);                         \
  MACRO(finalize, finalizer_function_t);

#define ALL_TYPES_NOSTATE(MACRO)                                               \
  ALL_TYPES_NOSTATE_BEGIN(MACRO)                                               \
  ALL_TYPES_NOSTATE_END(MACRO)

#define ALL_TYPES(MACRO)                                                       \
  ALL_TYPES_NOSTATE(MACRO)                                                     \
  MACRO(states, void *);


#ifdef __cplusplus

#define MEMBER_VECTOR(name, type) std::vector<type> name;

#define ACPP_TRACER_FUNCTION_VA_ARGS(type, ...)                                     \
  tracer_utils::tracer_state.call_##type(std::forward_as_tuple(__VA_ARGS__));

#define ACPP_TRACER_FUNCTION1ARG(type) ACPP_TRACER_FUNCTION_VA_ARGS(type)
#define ACPP_TRACER_FUNCTION2ARG(type, arg2) ACPP_TRACER_FUNCTION_VA_ARGS(type, arg2)
#define ACPP_TRACER_FUNCTION3ARG(type, arg2, arg3)                                  \
  ACPP_TRACER_FUNCTION_VA_ARGS(type, arg2, arg3)

#define INIT_FUNCTIONS(type, arg_type)                                         \
  void init_##type(arg_type);

#endif

#ifndef _WIN32
#define ACPP_COMMON_IMPORT
#else
#ifdef MYLIB_EXPORTS
#define ACPP_COMMON_IMPORT
#else
#define ACPP_COMMON_IMPORT __declspec(dllimport)
#endif
#endif

#endif // TRACER_MACROS_H
