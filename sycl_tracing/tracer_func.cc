#include "hipSYCL/sycl/tracer_utils.hpp"
#include <chrono>
#include <stdio.h>
#include <sycl/sycl.hpp>

using tracer_funcs = Tracer_utils::tracer_funcs;
using start_end = Tracer_utils::start_end;

// using time_point = Tracer_utils::time_point;

void tracer_func_submit(start_end) {
  std::cout << "Tracer function submit called" << std::endl;
}

void tracer_func_submit_secondary(start_end) {
  std::cout << "Tracer function submit_secondary called" << std::endl;
}

void tracer_func_parallel_for(start_end) {
  std::cout << "Tracer function parallel_for called" << std::endl;
}

void tracer_func_parallel_for_work_group(start_end) {
  std::cout << "Tracer function parallel_for_work_group called" << std::endl;
}

void tracer_func_single_task(start_end) {
  std::cout << "Tracer function single_task called" << std::endl;
}

void tracer_func_memcpy(start_end) {
  std::cout << "Tracer function memcpy called" << std::endl;
}

void tracer_func_wait(start_end) {
  std::cout << "Tracer function wait called" << std::endl;
}

void tracer_func_memset(start_end) {
  std::cout << "Tracer function memset called" << std::endl;
}

void tracer_func_finalize() {
  std::cout << "Finalize function called in destructor of runtime!"
            << std::endl;
}

void tracer_func_initializer(tracer_funcs &tracer_state) {
  std::cout << "Tracer function initializer called" << std::endl;
  tracer_state.submit.push_back(tracer_func_submit);
  tracer_state.submit_secondary.push_back(tracer_func_submit_secondary);
  tracer_state.parallel_for.push_back(tracer_func_parallel_for);
  tracer_state.parallel_for_work_group.push_back(
      tracer_func_parallel_for_work_group);
  tracer_state.single_task.push_back(tracer_func_single_task);
  tracer_state.memcpy.push_back(tracer_func_memcpy);
  tracer_state.wait.push_back(tracer_func_wait);
  tracer_state.memset.push_back(tracer_func_memset);
  tracer_state.finalize.insert(tracer_state.finalize.begin(),
                               tracer_func_finalize);
}
