#include "hipSYCL/sycl/tracer_utils.hpp"
#include <boost/container_hash/hash.hpp>
#include <boost/functional/hash.hpp>
#include <boost/stacktrace.hpp>
#include <cstdlib>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sycl/sycl.hpp>
#include <unordered_map>

using array_t = std::array<char, sizeof(sycl::event)>;

template <> struct std::hash<array_t> {
  std::size_t operator()(const array_t &arr) const {
    std::size_t result = 0;
    for (auto i : arr) {
      boost::hash_combine(result, i);
    }
    return result;
  }
};

#ifdef __cplusplus
extern "C" {
#endif

struct state_t {
  using event_map_t = std::unordered_map<array_t, std::string, std::hash<array_t>>;
  // std::unordered_map<sycl::queue, event_map_t> queue_map;
  event_map_t event_map;

  ~state_t() {

    // std::cout << "Hello World from the even leak destructor" << std::endl;
    std::cout << "The number of unfinished events is: " << event_map.size() << std::endl;
    for (auto &i : event_map) {
      std::cout << "Event not yet completed at the end of program" << std::endl;
    }
  }
};

state_t my_state;

void submit_end(void *usr_state, void *event_ptr, void *qptr) {
  sycl::event e = *((sycl::event *)event_ptr);
  std::cout << "submit end called!" << std::endl;
  boost::stacktrace::stacktrace st;

  std::stringstream ss;
  ss << st << std::endl;
  // std::cout << st << std::endl;

  array_t arr;
  std::memcpy(arr.data(), &e, sizeof(sycl::event));

  int i = 0;
  for (std::string line; std::getline(ss, line, '\n');) {
    if ((i == 3) | (i == 4)) {
      ((state_t *)usr_state)->event_map[arr] = line.substr(4, line.size() - 4);
    }
    i++;
  }
}

void wait_end(void *usr_state) {

  auto &map = ((state_t *)usr_state)->event_map;

  for (auto it = map.begin(); it != map.end();) {
    std::cout << "Hello World!" << std::endl;
    auto status = ((sycl::event *)(it->first.data()))
                      ->get_info<sycl::info::event::command_execution_status>();
    if (status == sycl::info::event_command_status::complete) {
      it = map.erase(it);
    } else {
      ++it;
    };
  }
}

void init_register() {
  state_t *state = &my_state;
  init_states(state);
  init_submit_end(submit_end);
  init_wait_end(wait_end);
}

#ifdef __cplusplus
}
#endif
