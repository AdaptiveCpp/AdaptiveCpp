#include "hipSYCL/sycl/tracer_utils.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <thread>

#ifdef __cplusplus
extern "C" {
#endif

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

std::ofstream outfile("outfile.json");

struct state_t {
  time_point start_timer;
  std::string type;
  std::ofstream outfile;
  int num_start = 0;
  int num_end = 0;
};

typedef state_t submission_state_t;
typedef state_t parallel_for_state_t;
typedef state_t parallel_for_work_group_state_t;
typedef state_t single_task_state_t;
typedef state_t wait_state_t;
typedef state_t memcpy_state_t;
typedef state_t memset_state_t;
typedef state_t copy_state_t;
typedef state_t fill_state_t;

void start(void *state_ptr) {
  state_t &state = *((state_t *)state_ptr);
  state.num_start++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_time - state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "B"},       {"tid", id_string},
                   {"pid", "0"},      {"name", state.type},
                   {"cat", "cpu_op"}, {"ts", duration_ns.count()},
                   {"id", 0}};

  outfile << a.dump() << "," << std::endl;

  std::cout << "Hello World from the " << state.type << "_start function!"
            << std::endl;
}

void end(void *state_ptr) {
  state_t &state = *((state_t *)state_ptr);
  state.num_end++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_time - state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "E"},       {"tid", id_string},
                   {"pid", "0"},      {"name", state.type},
                   {"cat", "cpu_op"}, {"ts", duration_ns.count()},
                   {"id", 0}};

  outfile << a.dump() << "," << std::endl;

  std::cout << "Hello World from the " << state.type << "_end function!"
            << std::endl;
}

void finalize(void *, void *, void *, void *, void *, void *, void *, void *,
              void *, void *) {
  outfile << "]}";
  outfile.close();
};

auto submission_start = start;
auto submission_end = end;
auto single_task_start = start;
auto single_task_end = end;
auto parallel_for_start = start;
auto parallel_for_end = end;
auto parallel_for_work_group_start = start;
auto parallel_for_work_group_end = end;
auto wait_start = start;
auto wait_end = end;
auto memcpy_start = start;
auto memcpy_end = end;
auto memset_start = start;
auto memset_end = end;
auto fill_start = start;
auto fill_end = end;
auto copy_start = start;
auto copy_end = end;

void init_register() {

  outfile << "{ \"traceEvents\": [" << std::endl;

  auto tracer_start_time = std::chrono::high_resolution_clock::now();

  submission_state_t *submission_state =
      new submission_state_t{tracer_start_time, "submission"};
  parallel_for_state_t *parallel_for_state =
      new parallel_for_state_t{tracer_start_time, "parallel_for"};
  single_task_state_t *single_task_state =
      new single_task_state_t{tracer_start_time, "single_task"};
  wait_state_t *wait_state = new wait_state_t{tracer_start_time, "wait"};
  memcpy_state_t *memcpy_state =
      new memcpy_state_t{tracer_start_time, "memcpy"};
  memset_state_t *memset_state =
      new memset_state_t{tracer_start_time, "memset"};
  parallel_for_work_group_state_t *parallel_for_work_group_state =
      new parallel_for_work_group_state_t{tracer_start_time,
                                          "parallel_for_work_group"};
  auto *copy_state = new copy_state_t{tracer_start_time, "copy"};
  auto *fill_state = new fill_state_t{tracer_start_time, "fill"};

  init_parallel_for_work_group_state(parallel_for_work_group_state);
  init_parallel_for_work_group_start(parallel_for_work_group_start);
  init_parallel_for_work_group_end(parallel_for_work_group_end);
  init_memset_state(memset_state);
  init_memset_start(memset_start);
  init_memset_end(memset_end);
  init_memcpy_state(memcpy_state);
  init_memcpy_start(memcpy_start);
  init_memcpy_end(memcpy_end);
  init_wait_state(wait_state);
  init_wait_start(wait_start);
  init_wait_end(wait_end);
  init_single_task_state(single_task_state);
  init_single_task_start(single_task_start);
  init_single_task_end(single_task_end);
  init_parallel_for_state(parallel_for_state);
  init_parallel_for_start(parallel_for_start);
  init_parallel_for_end(parallel_for_end);
  init_submit_state(submission_state);
  init_submit_start(submission_start);
  init_submit_end(submission_end);
  init_copy_state(copy_state);
  init_copy_start(copy_start);
  init_copy_end(copy_end);
  init_fill_state(fill_state);
  init_fill_start(fill_start);
  init_fill_end(fill_end);
  // init_finalizer(finalize);
}

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */
