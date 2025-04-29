#include "hipSYCL/sycl/tracer_utils.hpp"
#include <boost/stacktrace.hpp>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <set>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#ifdef __cplusplus
extern "C" {
#endif

struct state_t {
  std::unordered_map<void *, std::string> pointer_map;

  ~state_t() {
    std::unordered_map<std::string, void *> string_map;
    for (auto i : pointer_map) {
      string_map.insert({i.second, i.first});
    }

    for (auto i : string_map) {
      std::cout << "memory allocated at: " << std::endl;
      std::cout << i.first << std::endl;
      std::cout << "but never freed!" << std::endl;
    }
  }
};

state_t my_state;

auto malloc_device_start = [](void *usr_state) {
  std::cout << "Device pointer allocated" << std::endl;
};
auto malloc_device_end = [](void *usr_state, void *ptr) {
  // end(usr_state, 10, "malloc_device");
  std::cout << "malloc_device end called!" << std::endl;
  boost::stacktrace::stacktrace st;

  std::stringstream ss;
  ss << st << std::endl;

  int i = 0;
  for (std::string line; std::getline(ss, line, '\n');)
    if (i++ == 5) {
      ((state_t *)usr_state)->pointer_map[ptr] = line.substr(4, line.size() - 4);
    }
};

auto free_start = [](void *usr_state) {};

auto free_end = [](void *usr_state, void *ptr) { ((state_t *)usr_state)->pointer_map.erase(ptr); };

void init_register() {
  state_t *state = &my_state;

  init_state(state);
  init_malloc_device_start(malloc_device_start);
  init_malloc_device_end(malloc_device_end);
  init_malloc_shared_start(malloc_device_start);
  init_malloc_shared_end(malloc_device_end);
  init_malloc_host_start(malloc_device_start);
  init_malloc_host_end(malloc_device_end);

  init_free_start(free_start);
  init_free_end(free_end);
}

#ifdef __cplusplus
}
#endif
