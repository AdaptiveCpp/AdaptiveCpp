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
            (tracer_functs_initialize_t)dlsym(so_lib, "init_register");
        if (tracer_func_initializer) {
          tracer_func_initializer(tracer_state);
          set_tracer_equal_num(tracer_state);
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

void set_tracer_equal_num(tracer_funcs &tracer_state) {
  tracer_state.size++;

  if (tracer_state.submit_start.size() == tracer_state.size - 1) {
    tracer_state.submit_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_start.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.submit_end.size() == tracer_state.size - 1) {
    tracer_state.submit_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of submit_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.submit_state.size() == tracer_state.size - 1) {
    tracer_state.submit_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.submit_secondary_start.size() == tracer_state.size - 1) {
    tracer_state.submit_secondary_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_secondary_start.size() < tracer_state - 1) {
    std::cout
        << "Error: Number of submit_secondary_start function pointers smaller "
           "than number tracer files"
        << std::endl;
  }
#endif

  if (tracer_state.submit_secondary_end.size() == tracer_state.size - 1) {
    tracer_state.submit_secondary_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_secondary_end.size() < tracer_state - 1) {
    std::cout
        << "Error: Number of submit_secondary_end function pointers smaller "
           "than number tracer files"
        << std::endl;
  }
#endif

  if (tracer_state.submit_secondary_state.size() == tracer_state.size - 1) {
    tracer_state.submit_secondary_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.submit_secondary_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.parallel_for_start.size() == tracer_state.size - 1) {
    tracer_state.parallel_for_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_start.size() < tracer_state - 1) {
    std::cout
        << "Error: Number of parallel_for_start function pointers smaller "
           "than number tracer files"
        << std::endl;
  }
#endif

  if (tracer_state.parallel_for_end.size() == tracer_state.size - 1) {
    tracer_state.parallel_for_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of parallel_for_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.parallel_for_state.size() == tracer_state.size - 1) {
    tracer_state.parallel_for_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.parallel_for_work_group_start.size() ==
      tracer_state.size - 1) {
    tracer_state.parallel_for_work_group_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_work_group_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of parallel_for_work_group_start function "
                 "pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.parallel_for_work_group_end.size() ==
      tracer_state.size - 1) {
    tracer_state.parallel_for_work_group_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_work_group_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of parallel_for_work_group_end function "
                 "pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.parallel_for_work_group_state.size() ==
      tracer_state.size - 1) {
    tracer_state.parallel_for_work_group_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.parallel_for_work_group_state.size() <
      tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.single_task_start.size() == tracer_state.size - 1) {
    tracer_state.single_task_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.single_task_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of single_task_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.single_task_end.size() == tracer_state.size - 1) {
    tracer_state.single_task_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.single_task_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of single_task_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.single_task_state.size() == tracer_state.size - 1) {
    tracer_state.single_task_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.single_task_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memcpy_start.size() == tracer_state.size - 1) {
    tracer_state.memcpy_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memcpy_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of memcpy_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memcpy_end.size() == tracer_state.size - 1) {
    tracer_state.memcpy_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memcpy_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of memcpy_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memcpy_state.size() == tracer_state.size - 1) {
    tracer_state.memcpy_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memcpy_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.wait_start.size() == tracer_state.size - 1) {
    tracer_state.wait_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.wait_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of wait_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.wait_end.size() == tracer_state.size - 1) {
    tracer_state.wait_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.wait_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of wait_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.wait_state.size() == tracer_state.size - 1) {
    tracer_state.wait_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.wait_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memset_start.size() == tracer_state.size - 1) {
    tracer_state.memset_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memset_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of memset_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memset_end.size() == tracer_state.size - 1) {
    tracer_state.memset_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memset_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of memset_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.memset_state.size() == tracer_state.size - 1) {
    tracer_state.memset_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.memset_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.fill_start.size() == tracer_state.size - 1) {
    tracer_state.fill_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.fill_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of fill_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.fill_end.size() == tracer_state.size - 1) {
    tracer_state.fill_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.fill_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of fill_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.fill_state.size() == tracer_state.size - 1) {
    tracer_state.fill_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.fill_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.copy_start.size() == tracer_state.size - 1) {
    tracer_state.copy_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.copy_start.size() < tracer_state - 1) {
    std::cout << "Error: Number of copy_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.copy_end.size() == tracer_state.size - 1) {
    tracer_state.copy_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.copy_end.size() < tracer_state - 1) {
    std::cout << "Error: Number of copy_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (tracer_state.copy_state.size() == tracer_state.size - 1) {
    tracer_state.copy_state.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (tracer_state.copy_state.size() < tracer_state.size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif
}

void finalize_tracing() {
  for (auto func : tracer_state.finalize)
    func(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
         nullptr, nullptr);
}

} // namespace Tracer_utils
