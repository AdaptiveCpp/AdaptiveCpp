#include "hipSYCL/sycl/tracer_utils.hpp"
#include <boost/stacktrace.hpp>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <thread>
#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif

using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct state_t {
  time_point start_timer;
  std::ofstream outfile;

  std::unordered_map<void *, std::string> pointer_map;

  std::array<int, 13> num_starts{};
  std::array<int, 13> num_ends{};

  state_t(std::string outfile_name) : outfile(outfile_name) {
    std::cout << "Tracer state initialized" << std::endl;
  }

  ~state_t() {
    outfile << "]}";
    outfile.close();
    std::cout << "Tracing successfully finished" << std::endl;
  }
};

// static state_t my_state("outfile.json");

void start(void *state_ptr, int num, std::string type) {
  state_t &state = *((state_t *)state_ptr);
  state.num_starts[num]++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(start_time - state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "B"},    {"tid", id_string}, {"pid", "0"},
                   {"name", type}, {"cat", "cpu_op"},  {"ts", duration_micros.count()},
                   {"id", 0}};

  ((state_t *)state_ptr)->outfile << a.dump() << "," << std::endl;

  std::cout << "Hello World from the " << type << "_start function!" << std::endl;
}

void end(void *state_ptr, int num, std::string type) {
  state_t &state = *((state_t *)state_ptr);
  state.num_ends[num]++;

  auto start_time = std::chrono::high_resolution_clock::now();

  auto duration_micros =
      std::chrono::duration_cast<std::chrono::microseconds>(start_time - state.start_timer);

  std::string id_string;
  std::stringstream sid_string;
  sid_string << std::this_thread::get_id();

  nlohmann::json a{{"ph", "E"},    {"tid", id_string}, {"pid", "0"},
                   {"name", type}, {"cat", "cpu_op"},  {"ts", duration_micros.count()},
                   {"id", 0}};

  ((state_t *)state_ptr)->outfile << a.dump() << "," << std::endl;

  // std::cout << "Hello World from the " << type << "_end function!" << std::endl;
}

auto submission_start = [](void *usr_state) { start(usr_state, 0, "submission"); };
auto submission_end = [](void *usr_state, std::size_t evt, std::size_t ptr) {
  end(usr_state, 0, "submission");
};
auto single_task_start = [](void *usr_state) { start(usr_state, 1, "single_task"); };
auto single_task_end = [](void *usr_state) { end(usr_state, 1, "single_task"); };
auto parallel_for_start = [](void *usr_state) { start(usr_state, 2, "parallel_for"); };
auto parallel_for_end = [](void *usr_state) { end(usr_state, 2, "parallel_for"); };
auto parallel_for_work_group_start = [](void *usr_state) {
  start(usr_state, 3, "parallel_for_work_group");
};
auto parallel_for_work_group_end = [](void *usr_state) {
  end(usr_state, 3, "parallel_for_work_group");
};
auto wait_start = [](void *usr_state) { start(usr_state, 4, "wait"); };
auto wait_end = [](void *usr_state, std::size_t id) { end(usr_state, 4, "wait"); };
auto memcpy_start = [](void *usr_state) { start(usr_state, 5, "memcpy"); };
auto memcpy_end = [](void *usr_state) { end(usr_state, 5, "memcpy"); };
auto memset_start = [](void *usr_state) { start(usr_state, 6, "memset"); };
auto memset_end = [](void *usr_state) { end(usr_state, 6, "memset"); };
auto fill_start = [](void *usr_state) { start(usr_state, 7, "fill"); };
auto fill_end = [](void *usr_state) { end(usr_state, 7, "fill"); };
auto copy_start = [](void *usr_state) { start(usr_state, 8, "copy"); };
auto copy_end = [](void *usr_state) { end(usr_state, 8, "copy"); };

auto malloc_device_start = [](void *usr_state) { start(usr_state, 10, "malloc_device"); };
auto malloc_device_end = [](void *usr_state, void *ptr) { end(usr_state, 10, "malloc_device"); };

auto malloc_host_start = [](void *usr_state) { start(usr_state, 11, "malloc_host"); };
auto malloc_host_end = [](void *usr_state, void *ptr) { end(usr_state, 11, "malloc_host"); };

auto malloc_shared_start = [](void *usr_state) { start(usr_state, 12, "malloc_shared"); };
auto malloc_shared_end = [](void *usr_state, void *ptr) { end(usr_state, 12, "malloc_shared"); };

auto free_start = [](void *usr_state) { start(usr_state, 13, "sycl::free"); };
auto free_end = [](void *usr_state, void *ptr) { end(usr_state, 13, "sycl::free"); };

auto finalize = [](void *usr_state) { delete (state_t *)usr_state; };

void init_register() {

  auto tracer_start_time = std::chrono::high_resolution_clock::now();

  // my_state.start_timer = tracer_start_time;

  state_t *state = new state_t{"outfile.json"};

  state->start_timer = tracer_start_time;

  state->outfile << "{ \"traceEvents\": [" << std::endl;

  init_states(state);
  init_parallel_for_work_group_start(parallel_for_work_group_start);
  init_parallel_for_work_group_end(parallel_for_work_group_end);
  init_memset_start(memset_start);
  init_memset_end(memset_end);
  init_memcpy_start(memcpy_start);
  init_memcpy_end(memcpy_end);
  init_wait_start(wait_start);
  init_wait_event_end(wait_end);
  init_wait_queue_end(wait_end);
  init_single_task_start(single_task_start);
  init_single_task_end(single_task_end);
  init_parallel_for_start(parallel_for_start);
  init_parallel_for_end(parallel_for_end);
  init_submit_start(submission_start);
  init_submit_end(submission_end);
  init_copy_start(copy_start);
  init_copy_end(copy_end);
  init_fill_start(fill_start);
  init_fill_end(fill_end);
  init_malloc_device_start(malloc_device_start);
  init_malloc_device_end(malloc_device_end);
  init_malloc_host_start(malloc_host_start);
  init_malloc_host_end(malloc_host_end);
  init_malloc_shared_start(malloc_shared_start);
  init_malloc_shared_end(malloc_shared_end);
  init_free_start(free_start);
  init_free_end(free_end);
  init_finalize(finalize);
}

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */
