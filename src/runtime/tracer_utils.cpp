#include <dlfcn.h>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

#ifdef __cplusplus
extern "C" {
#endif

#define INIT_FUNCTION_DEFINITION(type, arg_type)                                                   \
  void init_##type(arg_type arg) { Tracer_utils::tracer_state.type.push_back(arg); }

ALL_TYPES(INIT_FUNCTION_DEFINITION);

#ifdef __cplusplus
}
#endif
