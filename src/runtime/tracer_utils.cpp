#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <sstream>
#include <unordered_map>

#include "hipSYCL/sycl/tracer_utils.hpp"

namespace Tracer_utils {
using time_point = std::chrono::high_resolution_clock::time_point;

// std::unordered_map<tracer_type, std::string> tracer_type_map{
//     {tracer_type::SUBMIT, "submit"},
//     {tracer_type::SUBMIT_SECONDARY, "submit_secondary"},
//     {tracer_type::PARALLEL_FOR, "parallel_for"},
//     {tracer_type::PARALLEL_FOR_WORK_GROUP, "parallel_reduce"},
//     {tracer_type::SINGLE_TASK, "single_task"},
//     {tracer_type::MEMCPY, "memcpy"},
//     {tracer_type::WAIT, "wait"}};
//
// std::unordered_map<tracer_type, std::size_t> Tracer_map{
//     {tracer_type::SUBMIT, 0},       {tracer_type::SUBMIT_SECONDARY, 0},
//     {tracer_type::PARALLEL_FOR, 0}, {tracer_type::PARALLEL_FOR_WORK_GROUP,
//     0}, {tracer_type::SINGLE_TASK, 0},  {tracer_type::MEMCPY, 0},
//     {tracer_type::WAIT, 0}};
//
// std::unordered_map<tracer_type, double> Tracer_time{
//     {tracer_type::SUBMIT, 0.0},
//     {tracer_type::SUBMIT_SECONDARY, 0.0},
//     {tracer_type::PARALLEL_FOR, 0.0},
//     {tracer_type::PARALLEL_FOR_WORK_GROUP, 0.0},
//     {tracer_type::SINGLE_TASK, 0.0},
//     {tracer_type::MEMCPY, 0.0},
//     {tracer_type::WAIT, 0.0}};

void (*tracer_func)(tracer_type, start_end) = nullptr;

tracer_func_t *tracer_funcs_array = nullptr;
int size = 0;

void initialize_tracers_from_env() {

  std::list<void *> so_libraries;
  std::list<tracer_func_t> tracer_funcs;

  if (const char *env_p = std::getenv("ACPP_TOOL_LIBRARY")) {
    std::string path(env_p);
    std::istringstream path_stream(path);

    for (std::string single_lib; std::getline(path_stream, single_lib, ':');) {
      // std::cout << "Library: " << single_lib << std::endl;

      void *so_lib = dlopen(single_lib.c_str(), RTLD_NOW | RTLD_LOCAL);

      if (so_lib) {
        // std::cout << "found library" << std::endl;
        so_libraries.push_back(so_lib);
        tracer_func_t tracer_func = (tracer_func_t)dlsym(so_lib, "tracer");
        if (tracer_func) {
          tracer_funcs.push_back(tracer_func);
        }
      }
    }

    if (tracer_funcs.size() > 0) {

      tracer_funcs_array = new tracer_func_t[tracer_funcs.size()];

      // Adjusting the tracker for the tracer_func_array size
      size = tracer_funcs.size();
      std::copy(tracer_funcs.begin(), tracer_funcs.end(), tracer_funcs_array);
    }
  }
}

void initialize_tracer(void (*func)(tracer_type, start_end)) {
  tracer_func = func;
}

void tracer_function(char *function_name, start_end state) {
  static auto start_time = std::chrono::high_resolution_clock::now();
  static auto end_time = std::chrono::high_resolution_clock::now();

  if (state == start_end::START) {
    start_time = std::chrono::high_resolution_clock::now();
  } else {
    end_time = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration<double>(end_time - start_time).count();

    std::cout << "Duration: " << duration << " seconds" << std::endl;
  }
}
} // namespace Tracer_utils
