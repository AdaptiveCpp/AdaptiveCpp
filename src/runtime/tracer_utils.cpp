#include <dlfcn.h>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

extern "C" void initialize_tracer(void (*func)(tracer_start_end),
                                  tracer_type type, void *tracer_state) {

  auto &state = *((Tracer_utils::tracer_funcs *)tracer_state);

  switch (type) {
  case SUBMIT:
    state.submit.push_back(func);
    break;
  case SUBMIT_SECONDARY:
    state.submit_secondary.push_back(func);
    break;
  case PARALLEL_FOR:
    state.parallel_for.push_back(func);
    break;
  case PARALLEL_FOR_WORK_GROUP:
    state.parallel_for_work_group.push_back(func);
    break;
  case SINGLE_TASK:
    state.single_task.push_back(func);
    break;
  case MEMCPY:
    state.memcpy.push_back(func);
    break;
  case WAIT:
    state.wait.push_back(func);
    break;
  case MEMSET:
    state.memset.push_back(func);
    break;
  case FILL:
    state.fill.push_back(func);
    break;
  case COPY:
    state.copy.push_back(func);
    break;
  case FINALIZE:
    state.finalize.insert(state.finalize.begin(), func);
    break;
  }
};
