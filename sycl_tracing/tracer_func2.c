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

struct submission_state_t {
  time_point start_timer;
  std::ofstream outfile;
  int num_start = 0;
  int num_end = 0;
};

struct parallel_for_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct parallel_for_work_group_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct single_task_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct wait_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct memcpy_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct memset_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct copy_sate_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct fill_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

struct copy_state_t {
  time_point start_timer;
  int num_start = 0;
  int num_end = 0;
};

void submission_start(void *submission_state_ptr) {
  submission_state_t &submission_state =
      *((submission_state_t *)submission_state_ptr);
  submission_state.num_start++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_time - submission_state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "B"},       {"tid", id_string},
                   {"pid", "0"},      {"name", "submission"},
                   {"cat", "cpu_op"}, {"ts", duration_ns.count()},
                   {"id", 0}};

  outfile << a.dump();

  std::cout << "Hello World from the submission_start function!" << std::endl;
}

void submission_end(void *submission_state_ptr) {
  submission_state_t &submission_state =
      *((submission_state_t *)submission_state_ptr);
  submission_state.num_end++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      start_time - submission_state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "X"},       {"tid", id_string},
                   {"pid", "0"},      {"name", "submission"},
                   {"cat", "cpu_op"}, {"ts", duration_ns.count()},
                   {"id", 0}};

  outfile << a.dump();

  std::cout << "Hello World from the submission_end function!" << std::endl;
}

// void submission_end(void *submission_state) {
//   ((submission_state_t *)submission_state)->num_end++;
//   std::cout << "Hello World from the submission_end function!" << std::endl;
// }

void single_task_start(void *single_task_state) {
  ((single_task_state_t *)single_task_state)->num_start++;
  std::cout << "Hello World from the single_task_start function!" << std::endl;
}

void single_task_end(void *single_task_state) {
  ((single_task_state_t *)single_task_state)->num_end++;
  std::cout << "Hello World from the single_task_end function!" << std::endl;
}

void parallel_for_start(void *parallel_for_state) {
  ((parallel_for_state_t *)parallel_for_state)->num_start++;
  std::cout << "Hello World from the parallel_for_start_function" << std::endl;
}

void parallel_for_end(void *parallel_for_state) {
  ((parallel_for_state_t *)parallel_for_state)->num_end++;
  std::cout << "Hello World from the parallel_for_end function" << std::endl;
}

void parallel_for_work_group_start(void *parallel_for_work_group_state) {
  ((parallel_for_work_group_state_t *)parallel_for_work_group_state)
      ->num_start++;
  std::cout << "Hello World from parallel_for_work_group_start function"
            << std::endl;
}

void parallel_for_work_group_end(void *parallel_for_work_group_state) {
  ((parallel_for_work_group_state_t *)parallel_for_work_group_state)->num_end++;
  std::cout << "Hello World from parallel_for_work_group_end function"
            << std::endl;
}

void wait_start(void *wait_state) {
  ((wait_state_t *)wait_state)->num_start++;
  std::cout << "Hello world from the wait_start funcion" << std::endl;
}

void wait_end(void *wait_state) {
  ((wait_state_t *)wait_state)->num_end++;
  std::cout << "Hello world from wait_end function" << std::endl;
}

void memcpy_start(void *memcpy_state) {
  ((memcpy_state_t *)memcpy_state)->num_start++;
  std::cout << "Hello world from the memcpy_start funcion" << std::endl;
}

void memcpy_end(void *memcpy_state) {
  ((memcpy_state_t *)memcpy_state)->num_end++;
  std::cout << "Hello world from memcpy_end function" << std::endl;
}

void memset_start(void *memset_state) {
  ((memset_state_t *)memset_state)->num_start++;
  std::cout << "Hello world from the memset_start funcion" << std::endl;
}

void memset_end(void *memset_state) {
  ((memset_state_t *)memset_state)->num_end++;
  std::cout << "Hello world from memset_end function" << std::endl;
}

void fill_start(void *fill_state) {
  ((fill_state_t *)fill_state)->num_start++;
  std::cout << "Hello World from inside the fill_start function" << std::endl;
}
void fill_end(void *fill_state) {
  ((fill_state_t *)fill_state)->num_end++;
  std::cout << "Hello World from inside the fill_end function" << std::endl;
}

void copy_start(void *copy_state) {
  ((copy_state_t *)copy_state)->num_start++;
  std::cout << "Hello World from inside the copy_start function" << std::endl;
}

void copy_end(void *copy_state) {
  ((copy_state_t *)copy_state)->num_end++;
  std::cout << "Hello World from inside the copy_end function" << std::endl;
}

void init_register() {

  auto tracer_start_time = std::chrono::high_resolution_clock::now();

  submission_state_t *submission_state =
      new submission_state_t{tracer_start_time};
  parallel_for_state_t *parallel_for_state =
      new parallel_for_state_t{tracer_start_time};
  single_task_state_t *single_task_state =
      new single_task_state_t{tracer_start_time};
  wait_state_t *wait_state = new wait_state_t{tracer_start_time};
  memcpy_state_t *memcpy_state = new memcpy_state_t{tracer_start_time};
  memset_state_t *memset_state = new memset_state_t{tracer_start_time};
  parallel_for_work_group_state_t *parallel_for_work_group_state =
      new parallel_for_work_group_state_t{tracer_start_time};
  auto *copy_state = new copy_state_t{tracer_start_time};
  auto *fill_state = new fill_state_t;

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
}

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */
