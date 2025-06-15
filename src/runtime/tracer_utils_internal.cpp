#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <sstream>
#include <unordered_map>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

namespace Tracer_utils {
using time_point = std::chrono::high_resolution_clock::time_point;

tracer_funcs tracer_state;
bool is_init = false;

void initialize_tracers_from_env() {

  if (is_init)
    return;

  // If the init has not run yet, we initialize and set the flag

  std::list<void *> so_libraries;

  if (const char *env_p = std::getenv("SYCL_TOOL_LIBRARY")) {
    std::string path(env_p);
    std::istringstream path_stream(path);

    for (std::string single_lib; std::getline(path_stream, single_lib, ':');) {
      // std::cout << "Library: " << single_lib << std::endl;

      void *so_lib = dlopen(single_lib.c_str(), RTLD_NOW | RTLD_LOCAL);

      if (so_lib) {
        // std::cout << "found library" << std::endl;
        so_libraries.push_back(so_lib);
        tracer_functs_initialize_t tracer_func_initializer =
            (tracer_functs_initialize_t)dlsym(so_lib,
                                              "tracer_func_initializer");
        if (tracer_func_initializer) {
          tracer_func_initializer(tracer_state);
        } else {
          std::cerr << "Could not find "
                       "void tracer_func_initializer(start_end) in "
                       "library "
                    << single_lib << std::endl;
        }
      }
    }
  }

  is_init = true;
}

void finalize_tracing() {
  for (auto func : tracer_state.finalize)
    func(END);
}

} // namespace Tracer_utils
