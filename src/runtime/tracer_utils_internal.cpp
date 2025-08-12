#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <sstream>
#include <unordered_map>

#include "hipSYCL/sycl/tracer_utils.hpp"
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

namespace Tracer_utils {
using time_point = std::chrono::high_resolution_clock::time_point;

bool is_init = false;

void tracer_funcs::set_tracer_equal_num() {
  this->size++;

  if (this->submit_start.size() == this->size - 1) {
    this->submit_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->submit_start.size() < this->size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->submit_end.size() == this->size - 1) {
    this->submit_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->submit_end.size() < this->- 1) {
    std::cout << "Error: Number of submit_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->submit_secondary_start.size() == this->size - 1) {
    this->submit_secondary_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->submit_secondary_start.size() < this->- 1) {
    std::cout << "Error: Number of submit_secondary_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->submit_secondary_end.size() == this->size - 1) {
    this->submit_secondary_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->submit_secondary_end.size() < this->- 1) {
    std::cout << "Error: Number of submit_secondary_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->parallel_for_start.size() == this->size - 1) {
    this->parallel_for_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->parallel_for_start.size() < this->- 1) {
    std::cout << "Error: Number of parallel_for_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->parallel_for_end.size() == this->size - 1) {
    this->parallel_for_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->parallel_for_end.size() < this->- 1) {
    std::cout << "Error: Number of parallel_for_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->parallel_for_work_group_start.size() == this->size - 1) {
    this->parallel_for_work_group_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->parallel_for_work_group_start.size() < this->- 1) {
    std::cout << "Error: Number of parallel_for_work_group_start function "
                 "pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->parallel_for_work_group_end.size() == this->size - 1) {
    this->parallel_for_work_group_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->parallel_for_work_group_end.size() < this->- 1) {
    std::cout << "Error: Number of parallel_for_work_group_end function "
                 "pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->single_task_start.size() == this->size - 1) {
    this->single_task_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->single_task_start.size() < this->- 1) {
    std::cout << "Error: Number of single_task_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->single_task_end.size() == this->size - 1) {
    this->single_task_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->single_task_end.size() < this->- 1) {
    std::cout << "Error: Number of single_task_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->memcpy_start.size() == this->size - 1) {
    this->memcpy_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->memcpy_start.size() < this->- 1) {
    std::cout << "Error: Number of memcpy_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->memcpy_end.size() == this->size - 1) {
    this->memcpy_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->memcpy_end.size() < this->- 1) {
    std::cout << "Error: Number of memcpy_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->wait_start.size() == this->size - 1) {
    this->wait_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->wait_start.size() < this->- 1) {
    std::cout << "Error: Number of wait_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->wait_end.size() == this->size - 1) {
    this->wait_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->wait_end.size() < this->- 1) {
    std::cout << "Error: Number of wait_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->memset_start.size() == this->size - 1) {
    this->memset_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->memset_start.size() < this->- 1) {
    std::cout << "Error: Number of memset_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->memset_end.size() == this->size - 1) {
    this->memset_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->memset_end.size() < this->- 1) {
    std::cout << "Error: Number of memset_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->fill_start.size() == this->size - 1) {
    this->fill_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->fill_start.size() < this->- 1) {
    std::cout << "Error: Number of fill_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->fill_end.size() == this->size - 1) {
    this->fill_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->fill_end.size() < this->- 1) {
    std::cout << "Error: Number of fill_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->copy_start.size() == this->size - 1) {
    this->copy_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->copy_start.size() < this->- 1) {
    std::cout << "Error: Number of copy_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->copy_end.size() == this->size - 1) {
    this->copy_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->copy_end.size() < this->- 1) {
    std::cout << "Error: Number of copy_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->finalize.size() == this->size - 1) {
    this->finalize.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->finalize.size() < this->size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->states.size() == this->size - 1) {
    this->states.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->states.size() < this->size - 1) {
    std::cout << "Error: Number of submit start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_device_start.size() == this->size - 1) {
    this->malloc_device_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_device_start.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_start_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_shared_start.size() == this->size - 1) {
    this->malloc_shared_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_shared_start.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_shared_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_host_start.size() == this->size - 1) {
    this->malloc_host_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_host_start.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_host_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_device_end.size() == this->size - 1) {
    this->malloc_device_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_device_end.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_end_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_shared_end.size() == this->size - 1) {
    this->malloc_shared_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_shared_end.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_shared_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->malloc_host_end.size() == this->size - 1) {
    this->malloc_host_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->malloc_host_end.size() < this->size - 1) {
    std::cout << "Error: Number of malloc_host_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->free_end.size() == this->size - 1) {
    this->free_end.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->free_end.size() < this->size - 1) {
    std::cout << "Error: Number of free_end function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif

  if (this->free_start.size() == this->size - 1) {
    this->free_start.push_back(nullptr);
  }

#ifdef DEBUG_TRACER_LEVEL
  if (this->free_start.size() < this->size - 1) {
    std::cout << "Error: Number of free_start function pointers smaller "
                 "than number tracer files"
              << std::endl;
  }
#endif
}

tracer_funcs::tracer_funcs() {

  // If the init has not run yet, we initialize and set the flag

  std::cout << "Hello World from inside the tracer_funcs constructor" << std::endl;

  std::list<void *> so_libraries;

  if (const char *env_p = std::getenv("SYCL_TOOL_LIBRARY")) {
    std::string path(env_p);
    std::istringstream path_stream(path);

    for (std::string single_lib; std::getline(path_stream, single_lib, ':');) {
      // std::cout << "Library: " << single_lib << std::endl;

      void *so_lib = dlopen(single_lib.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);

      if (so_lib) {
        // std::cout << "found library" << std::endl;
        so_libraries.push_back(so_lib);
        tracer_functs_initialize_t tracer_func_initializer =
            (tracer_functs_initialize_t)dlsym(so_lib, "init_register");
        if (tracer_func_initializer) {
          tracer_func_initializer();
          this->set_tracer_equal_num();
        } else {
          std::cerr << "Could not find "
                       "void tracer_func_initializer(start_end) in "
                       "library "
                    << single_lib << std::endl;
        }
      } else {
        std::cout << "Warning: could not find library " << single_lib << std::endl;
        std::cerr << dlerror() << std::endl;
      }
    }
  }
}

tracer_funcs::~tracer_funcs() {
  for (int i = this->size - 1; i >= 0; i--)
    if (this->finalize[i] != nullptr)
      this->finalize[i](this->states[i]);
}

tracer_funcs tracer_state;

} // namespace Tracer_utils
