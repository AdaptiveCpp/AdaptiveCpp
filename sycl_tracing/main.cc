#include <chrono>
#include <iostream>
#include <map>
#include <unordered_map>

using time_point = std::chrono::high_resolution_clock::time_point;

namespace Tracer_utils {

using time_point = std::chrono::high_resolution_clock::time_point;

enum class start_end { START = 0, END = 1 };

enum class tracer_type {
  SUBMIT = 0,
  PARALLEL_FOR = 1,
  PARALLEL_REDUCE = 2,
  SINGLE_TASK = 3,
  MEMCPY = 4,
  WAIT = 5,
};

extern std::unordered_map<tracer_type, std::string> tracer_type_map;

std::map<tracer_type, std::size_t> Tracer_map;

std::map<tracer_type, double> Tracer_time;

extern void (*tracer_func)(tracer_type, start_end);

void initialize_tracer(void (*func)(tracer_type, start_end));

void tracer_function(char *function_name, start_end state);
}; // namespace Tracer_utils

time_point start_time;
time_point end_time;

int main() {
  int a = 0;
  Tracer_utils::start_end state = static_cast<Tracer_utils::start_end>(a);

  std::cout << "Hello, World!" << std::endl;
  return 0;
}
