#include <chrono>
#include <iostream>
#include <map>
#include <unordered_map>

#include "tracer_utils.hpp"

namespace Tracer_utils {
using time_point = std::chrono::high_resolution_clock::time_point;

std::unordered_map<tracer_type, std::string> tracer_type_map{
    {tracer_type::SUBMIT, "submit"},
    {tracer_type::PARALLEL_FOR, "parallel_for"},
    {tracer_type::PARALLEL_REDUCE, "parallel_reduce"},
    {tracer_type::SINGLE_TASK, "single_task"},
    {tracer_type::MEMCPY, "memcpy"},
    {tracer_type::WAIT, "wait"}};

std::map<tracer_type, std::size_t> Tracer_map{
    {tracer_type::SUBMIT, 0},          {tracer_type::PARALLEL_FOR, 0},
    {tracer_type::PARALLEL_REDUCE, 0}, {tracer_type::SINGLE_TASK, 0},
    {tracer_type::MEMCPY, 0},          {tracer_type::WAIT, 0}};

std::map<tracer_type, double> Tracer_time{
    {tracer_type::SUBMIT, 0.0},          {tracer_type::PARALLEL_FOR, 0.0},
    {tracer_type::PARALLEL_REDUCE, 0.0}, {tracer_type::SINGLE_TASK, 0.0},
    {tracer_type::MEMCPY, 0.0},          {tracer_type::WAIT, 0.0}};

void (*tracer_func)(tracer_type, start_end) = nullptr;

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
