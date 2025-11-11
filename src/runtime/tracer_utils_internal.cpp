#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <list>
#include <sstream>
#include <unordered_map>

#include "hipSYCL/common/dylib_loader.hpp"
#include "hipSYCL/sycl/tracer_utils.hpp"

#define MYLIB_EXPORTS
#include "hipSYCL/sycl/tracer_utils_internal.hpp"

#define EQUALIZE_HELPER(type)                                                                      \
  if (this->type.size() == this->size - 1) {                                                       \
    this->type.push_back(nullptr);                                                                 \
  }

#define DEBUG(type)                                                                                \
  if (this->type.size() < this->size - 1) {                                                        \
    std::cout << "Error: Number of " << #type                                                      \
              << " function pointers smaller than number tracer files" << std::endl;               \
  }

#define CLEAR(type, x) this->type.clear();

#ifdef DEBUG_TRACER_LEVEL
#define EQUALIZER(type) EQUALIZE_HELPER(type) DEBUG(type)
#else
#define EQUALIZER(type, x) EQUALIZE_HELPER(type)
#endif

namespace Tracer_utils {
using time_point = std::chrono::high_resolution_clock::time_point;

bool is_init = false;

void tracer_funcs::set_tracer_equal_num() {
  this->size++;
  ALL_TYPES(EQUALIZER);
}

std::list<void *> so_libraries;
std::list<tracer_functs_initialize_t> init_funcs;

void tracer_funcs::initialize_tracer() {

  // If the init has not run yet, we initialize and set the flag

  std::cout << "Hello World from inside the tracer_funcs constructor" << std::endl;

  if (const char *env_p = std::getenv("SYCL_TOOL_LIBRARY")) {
    std::string path(env_p);
    std::istringstream path_stream(path);

    for (std::string single_lib; std::getline(path_stream, single_lib, ':');) {
      // std::cout << "Library: " << single_lib << std::endl;
      std::string message{};
      void *so_lib = hipsycl::common::load_library(single_lib, message);
      if (!message.empty())
        std::cout << message << std::endl;

      if (so_lib) {
        // std::cout << "found library" << std::endl;
        std::string message{};
        so_libraries.push_back(so_lib);
        tracer_functs_initialize_t tracer_func_initializer =
            (tracer_functs_initialize_t)hipsycl::common::get_symbol_from_library(
                so_lib, "init_register", message);

        if (!message.empty())
          std::cout << message << std::endl;

        init_funcs.push_back(tracer_func_initializer);

        if (tracer_func_initializer) {
          tracer_func_initializer();
          this->set_tracer_equal_num();
        }
      }
    }
  }
}

void tracer_funcs::run_finalizers() {

  std::cout << "Hello World from inside the tracer_funcs finalizer stuff" << std::endl;
  for (int i = this->size - 1; i >= 0; i--)
    if (this->finalize[i] != nullptr)
      this->finalize[i](this->states[i]);

  clear_all();
}

void tracer_funcs::clear_all() {
  this->size = 0;
  ALL_TYPES(CLEAR);
}

ACPP_COMMON_EXPORT tracer_funcs tracer_state;
} // namespace Tracer_utils
