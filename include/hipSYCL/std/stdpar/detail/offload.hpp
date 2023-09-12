/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_PSTL_OFFLOAD_HPP
#define HIPSYCL_PSTL_OFFLOAD_HPP

#include "hipSYCL/std/stdpar/detail/execution_fwd.hpp"
#include "hipSYCL/std/stdpar/detail/stdpar_builtins.hpp"
#include "hipSYCL/std/stdpar/detail/sycl_glue.hpp"
#include <iterator>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <limits>
#include <utility>

namespace hipsycl::stdpar {

namespace algorithm_type {
struct for_each {};
struct for_each_n {};
struct transform {};
struct copy {};
struct copy_if {};
struct copy_n {};
struct fill {};
struct fill_n {};
struct generate {};
struct generate_n {};
struct replace {};
struct replace_if {};
struct replace_copy {};
struct replace_copy_if {};
struct find {};
struct find_if {};
struct find_if_not {};
struct all_of {};
struct any_of {};
struct none_of {};

struct transform_reduce {};
struct reduce {};
} // namespace algorithm_type

class offload_heuristic {
public:
  offload_heuristic() {
    auto devs = sycl::device::get_devices();
    auto offloading_backend = hipsycl::stdpar::detail::single_device_dispatch::get_queue()
              .get_device()
              .get_backend();
    for(const auto& dev : devs) {
      if (dev.get_backend() == offloading_backend) {
        sycl::queue q{dev, hipsycl::sycl::property_list{
                               hipsycl::sycl::property::queue::in_order{},
                               hipsycl::sycl::property::queue::
                                   hipSYCL_coarse_grained_events{}}};
        std::size_t s = measure_crossover_problem_size(q);
        _offloading_min_problem_sizes.push_back(std::make_pair(dev, s));
      } else {
        _offloading_min_problem_sizes.push_back(std::make_pair(dev, 0));
      }
    }

    for(const auto& v : _offloading_min_problem_sizes) {
      HIPSYCL_DEBUG_INFO
        << "PSTL offloading to device "
        << v.first.template get_info<sycl::info::device::name>()
        << " will be enabled for problem sizes > " << v.second << std::endl;
    }
  }

  // TODO: Currently, this conducts performance measurements when first invoked.
  // We should add save the results to disk, and only rerun benchmarks if no
  // previous results are found in the filesystem.
  // Also, note that this heuristic focuses entirely on the problem size,
  // we do not take into account the impact of potential data transfers.
  static std::size_t get_offloading_min_problem_size(const sycl::device &dev) {
    static offload_heuristic h;

    for(const auto& v : h._offloading_min_problem_sizes) {
      if(v.first == dev)
        return v.second;    
    }

    return 0;
  }

private:
  template<class F>
  double measure_time(int num_times, F&& f) {
    f(); // To exclude data transfers and JIT
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_times; ++i)
      f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return (static_cast<double>(duration.count()) * 1.e-9) / num_times;
  }

  template<class F>
  double measure_time(F&& f) {
    auto start = std::chrono::high_resolution_clock::now();
    f();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    return static_cast<double>(duration.count()) * 1.e-9;
  }

  template<class Queue>
  void run_offload(Queue& q, float* data, std::size_t problem_size){
    q.parallel_for(sycl::range<1>{problem_size}, [=](auto idx) {
      auto &x = data[idx];

      x *= x;
      x += 1;
    });
    q.wait();
  }

  void run_host(float* data, std::size_t problem_size){
    std::for_each(par_unseq_host_fallback, data, data + problem_size,
                  [=](auto &x) {
                    x *= x;
                    x += 1;
                  });
  }

  template<class Queue>
  std::size_t measure_crossover_problem_size(Queue& q) {
    constexpr std::size_t max_size = 1024*1024*128;
    constexpr std::size_t min_measurement_size = 32*1024*1024;
    float* device_data = sycl::malloc_shared<float>(max_size, q);
    float* host_data = sycl::malloc_shared<float>(max_size, q);

    std::memset(host_data, 2, max_size * sizeof(float));
    q.parallel_for(sycl::range<1>{max_size}, [=](auto idx){
      device_data[idx] = 2;
    });
    q.wait();
    
    std::size_t current_size = 8192;
    for(; current_size <= max_size; current_size *= 2) {
      std::size_t num_measurements =
          std::max(std::size_t{1}, min_measurement_size / current_size);

      double t_offload = measure_time(num_measurements, [&](){
        run_offload(q, device_data, current_size);
      });
      double t_host = measure_time(num_measurements, [&](){
        run_host(host_data, current_size);
      });
      
      if(t_offload < t_host)
        break;
    }

    sycl::free(device_data, q);
    sycl::free(host_data, q);
    
    return current_size;
  }

  std::vector<std::pair<sycl::device, std::size_t>> _offloading_min_problem_sizes;
};

template <class AlgorithmType, class Size, typename... Args>
bool should_offload(AlgorithmType type, Size n, const Args &...args) {
#ifdef __HIPSYCL_STDPAR_UNCONDITIONAL_OFFLOAD__
  return true;
#else
  auto& q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  std::size_t min_size =
      offload_heuristic::get_offloading_min_problem_size(q.get_device());
  
  return n > min_size;
#endif
}

#define HIPSYCL_STDPAR_OFFLOAD_NORET(algorithm_type_object, problem_size,      \
                                     offload_invoker, fallback_invoker, ...)   \
  __hipsycl_stdpar_consume_sync();                                             \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::should_offload(                         \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  if (is_offloaded) {                                                          \
    offload_invoker(q);                                                        \
  } else {                                                                     \
    fallback_invoker();                                                        \
  }                                                                            \
  __hipsycl_stdpar_optimizable_sync(q, is_offloaded);

#define HIPSYCL_STDPAR_OFFLOAD(algorithm_type_object, problem_size,            \
                               return_type, offload_invoker, fallback_invoker, \
                               ...)                                            \
  __hipsycl_stdpar_consume_sync();                                             \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::should_offload(                         \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  return_type ret = is_offloaded ? offload_invoker(q) : fallback_invoker();    \
  __hipsycl_stdpar_optimizable_sync(q, is_offloaded);                          \
  return ret;

#define HIPSYCL_STDPAR_BLOCKING_OFFLOAD(algorithm_type_object, problem_size,   \
                                        return_type, offload_invoker,          \
                                        fallback_invoker, ...)                 \
  __hipsycl_stdpar_consume_sync();                                             \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::should_offload(                         \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  const auto blocking_fallback_invoker = [&]() {                               \
    q.wait();                                                                  \
    return fallback_invoker();                                                 \
  };                                                                           \
  return_type ret =                                                            \
      is_offloaded ? offload_invoker(q) : blocking_fallback_invoker();         \
  return ret;
} // namespace hipsycl::stdpar

#endif
