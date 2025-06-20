#include <dlfcn.h>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// void initialize_tracer(void (*func)(tracer_start_end), tracer_type type,
//                        void *tracer_state) {
//
//   auto &state = *((Tracer_utils::tracer_funcs *)tracer_state);
//
//   switch (type) {
//   case SUBMIT:
//     state.submit.push_back(func);
//     break;
//   case SUBMIT_SECONDARY:
//     state.submit_secondary.push_back(func);
//     break;
//   case PARALLEL_FOR:
//     state.parallel_for.push_back(func);
//     break;
//   case PARALLEL_FOR_WORK_GROUP:
//     state.parallel_for_work_group.push_back(func);
//     break;
//   case SINGLE_TASK:
//     state.single_task.push_back(func);
//     break;
//   case MEMCPY:
//     state.memcpy.push_back(func);
//     break;
//   case WAIT:
//     state.wait.push_back(func);
//     break;
//   case MEMSET:
//     state.memset.push_back(func);
//     break;
//   case FILL:
//     state.fill.push_back(func);
//     break;
//   case COPY:
//     state.copy.push_back(func);
//     break;
//   case FINALIZE:
//     state.finalize.insert(state.finalize.begin(), func);
//     break;
//   }
// };

// Defining the end initialier functions
void init_submit_state(void *usr_state) {
  Tracer_utils::tracer_state.submit_state.push_back(usr_state);
}

void init_submit_secondary_state(void *usr_state) {
  Tracer_utils::tracer_state.submit_secondary_state.push_back(usr_state);
}

void init_parallel_for_state(void *usr_state) {
  Tracer_utils::tracer_state.parallel_for_state.push_back(usr_state);
}

void init_parallel_for_work_group_state(void *usr_state) {
  Tracer_utils::tracer_state.parallel_for_work_group_state.push_back(usr_state);
}

void init_single_task_state(void *usr_state) {
  Tracer_utils::tracer_state.single_task_state.push_back(usr_state);
}

void init_memcpy_state(void *usr_state) {
  Tracer_utils::tracer_state.memcpy_state.push_back(usr_state);
}

void init_wait_state(void *usr_state) {
  Tracer_utils::tracer_state.wait_state.push_back(usr_state);
}

void init_memset_state(void *usr_state) {
  Tracer_utils::tracer_state.memset_state.push_back(usr_state);
}

void init_fill_state(void *usr_state) {
  Tracer_utils::tracer_state.fill_state.push_back(usr_state);
}

void init_copy_state(void *usr_state) {
  Tracer_utils::tracer_state.copy_state.push_back(usr_state);
}

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

// Defining the initializers for the end functions
void init_submit_end(tracer_function_t usr_func) {
  Tracer_utils::tracer_state.submit_end.push_back(usr_func);
}

void init_submit_secondary_end(tracer_function_t usr_func) {
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

void init_finalizer(finalizer_function_t usr_func) {
  Tracer_utils::tracer_state.finalize.push_back(usr_func);
}

#ifdef __cplusplus
}
#endif
