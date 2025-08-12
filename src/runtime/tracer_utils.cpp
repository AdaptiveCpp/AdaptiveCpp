#include <dlfcn.h>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// Defining the end initialier functions
void init_state(void *usr_state) { Tracer_utils::tracer_state.states.push_back(usr_state); }

// Defining the start initializer functions
void init_submit_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.submit_start.push_back(usr_func);
}

void init_submit_secondary_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.submit_secondary_start.push_back(usr_func);
}

void init_parallel_for_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.parallel_for_start.push_back(usr_func);
}

void init_parallel_for_work_group_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.parallel_for_work_group_start.push_back(usr_func);
}

void init_single_task_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.single_task_start.push_back(usr_func);
}

void init_memcpy_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.memcpy_start.push_back(usr_func);
}

void init_wait_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.wait_start.push_back(usr_func);
}

void init_memset_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.memset_start.push_back(usr_func);
}

void init_fill_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.fill_start.push_back(usr_func);
}

void init_copy_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.copy_start.push_back(usr_func);
}

void init_malloc_host_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_host_start.push_back(usr_func);
}

void init_malloc_shared_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_shared_start.push_back(usr_func);
}

void init_malloc_device_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_device_start.push_back(usr_func);
}

// Defining the initializers for the end functions
void init_submit_end(tracer_function_submit_t usr_func) {
  Tracer_utils::tracer_state.submit_end.push_back(usr_func);
}

void init_submit_secondary_end(tracer_function_submit_t usr_func) {
  Tracer_utils::tracer_state.submit_secondary_end.push_back(usr_func);
}

void init_parallel_for_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.parallel_for_end.push_back(usr_func);
}

void init_parallel_for_work_group_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.parallel_for_work_group_end.push_back(usr_func);
}

void init_single_task_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.single_task_end.push_back(usr_func);
}

void init_memcpy_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.memcpy_end.push_back(usr_func);
}

void init_wait_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.wait_end.push_back(usr_func);
}

void init_memset_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.memset_end.push_back(usr_func);
}

void init_fill_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.fill_end.push_back(usr_func);
}

void init_copy_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.copy_end.push_back(usr_func);
}

void init_malloc_host_end(malloc_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_host_end.push_back(usr_func);
}

void init_malloc_shared_end(malloc_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_shared_end.push_back(usr_func);
}

void init_malloc_device_end(malloc_function_t usr_func) {
  Tracer_utils::tracer_state.malloc_device_end.push_back(usr_func);
}

void init_free_start(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.free_start.push_back(usr_func);
}

void init_free_end(malloc_function_t usr_func) {
  Tracer_utils::tracer_state.free_end.push_back(usr_func);
}

void init_finalizer(finalizer_function_t usr_func) {
  Tracer_utils::tracer_state.finalize.push_back(usr_func);
}

#ifdef __cplusplus
}
#endif
