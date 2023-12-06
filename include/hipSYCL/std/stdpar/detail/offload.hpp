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

#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/std/stdpar/detail/execution_fwd.hpp"
#include "hipSYCL/std/stdpar/detail/stdpar_builtins.hpp"
#include "hipSYCL/std/stdpar/detail/sycl_glue.hpp"
#include "hipSYCL/glue/reflection.hpp"
#include <atomic>
#include <cstring>
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

template<class T, typename... Args>
struct decorated_type {
  __attribute__((always_inline))
  decorated_type(const T& arg)
  : value{arg} {}

  template<class Decoration>
  static constexpr bool has_decoration() {
    return (std::is_same_v<Decoration, Args> || ...);
  }

  using value_type = T;
  T value;
};

template<class Decoration, class T>
constexpr bool has_decoration(const T& x) {
  return false;
}

template<class Decoration, typename... Args>
constexpr bool has_decoration(const decorated_type<Args...> &x) {
  return decorated_type<Args...>::template has_decoration<Decoration>();
}

namespace decorations {
struct no_pointer_validation {};
}

template<class T, typename... Attributes>
__attribute__((always_inline))
auto decorate(const T& x, Attributes... attrs) {
  return decorated_type<T, Attributes...>{x};
}

#define HIPSYCL_STDPAR_DECORATE(Arg, ...)                                      \
  hipsycl::stdpar::decorate(Arg, __VA_ARGS__)
#define HIPSYCL_STDPAR_NO_PTR_VALIDATION(Arg)                                  \
  HIPSYCL_STDPAR_DECORATE(                                                     \
      Arg, hipsycl::stdpar::decorations::no_pointer_validation{})

namespace detail {

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

template<class T>
bool validate_contained_pointers(const T& x) {
  if(has_decoration<decorations::no_pointer_validation>(x)) {
    return true;
  }

  auto& q = detail::single_device_dispatch::get_queue();
  auto* allocator = q.get_context()
      .hipSYCL_runtime()
      ->backends()
      .get(q.get_device().get_backend())
      ->get_allocator(q.get_device().hipSYCL_device_id());

  // Check if all contained pointers are valid on the device; otherwise
  // return false.
  glue::reflection::introspect_flattened_struct introspection{x};
  for(int i = 0; i < introspection.get_num_members(); ++i) {
    if(introspection.get_member_kind(i) == glue::reflection::type_kind::pointer) {
      void* ptr = nullptr;
      std::memcpy(&ptr, (char*)&x + introspection.get_member_offset(i), sizeof(void*));
      // Ignore nullptr
      if(ptr) {
        rt::pointer_info pinfo;
        if(!allocator->query_pointer(ptr, pinfo).is_success())
          return false;
      }
    }
  }
  return true;
}

template<typename... Args>
bool validate_all_pointers(const Args&... args){
  bool result = true;
  
  auto f = [&](const auto& arg){
    bool ptr_validation = validate_contained_pointers(arg);
    result = result && ptr_validation;
  };
  (f(args), ...);
  return result;
}

enum prefetch_mode {
  automatic = 0,
  always = 1,
  never = 2,
  after_sync = 3,
  first = 4
};

inline constexpr prefetch_mode get_prefetch_mode() noexcept {
#ifdef __HIPSYCL_STDPAR_PREFETCH_MODE__
  prefetch_mode mode = static_cast<prefetch_mode>(__HIPSYCL_STDPAR_PREFETCH_MODE__);
#else
  prefetch_mode mode = prefetch_mode::automatic;
#endif
  return mode;
}

inline void prefetch(sycl::queue& q, const void* ptr, std::size_t bytes) noexcept {
  auto* inorder_executor = q.hipSYCL_inorder_executor();
  if(inorder_executor) {
    // Attempt to invoke backend functionality directly -
    // in general we might have to issue multiple prefetches for
    // each kernel, so overheads can quickly add up.
    HIPSYCL_DEBUG_INFO << "[stdpar] Submitting raw prefetch to backend: "
                       << bytes << " bytes @" << ptr << std::endl;
    rt::inorder_queue* ordered_q = inorder_executor->get_queue();
    rt::prefetch_operation op{ptr, bytes, ordered_q->get_device()};
    ordered_q->submit_prefetch(op, nullptr);
  } else {
    q.prefetch(ptr, bytes);
  }
}

template<class AlgorithmType, class Size, typename... Args>
void prepare_offloading(AlgorithmType type, Size problem_size, const Args&... args) {
  auto& q = detail::single_device_dispatch::get_queue();
  std::size_t current_batch_id = stdpar::detail::stdpar_tls_runtime::get()
                                     .get_current_offloading_batch_id();

  
  // Use "first" mode in case of automatic prefetch decision for now
  const auto prefetch_mode =
      (get_prefetch_mode() == prefetch_mode::automatic) ? prefetch_mode::first
                                                        : get_prefetch_mode();

  auto prefetch_handler = [&](const auto& arg){
    if(!has_decoration<decorations::no_pointer_validation>(arg)) {
      glue::reflection::introspect_flattened_struct introspection{arg};
      for(int i = 0; i < introspection.get_num_members(); ++i) {
        if(introspection.get_member_kind(i) == glue::reflection::type_kind::pointer) {
          void* ptr = nullptr;
          std::memcpy(&ptr, (char*)&arg + introspection.get_member_offset(i), sizeof(void*));
          
          unified_shared_memory::allocation_lookup_result lookup_result;
          if(ptr && unified_shared_memory::allocation_lookup(ptr, lookup_result)) {
            int64_t *most_recent_offload_batch_ptr =
                &(lookup_result.info->most_recent_offload_batch);

            std::size_t prefetch_size = lookup_result.info->allocation_size;

            // Need to use atomic builtins until we can use C++ 20 atomic_ref :(
            std::size_t most_recent_offload_batch = __atomic_load_n(
                most_recent_offload_batch_ptr, __ATOMIC_ACQUIRE);
            
            bool should_prefetch = false;
            if(prefetch_mode == prefetch_mode::first)
              // an allocation that was never used will still contain the
              // initialization value of -1
              should_prefetch = most_recent_offload_batch == -1;
            else
              // Never emit multiple prefetches for the same allocation in one batch
              should_prefetch = most_recent_offload_batch < current_batch_id;

            if (should_prefetch) {
              prefetch(q, lookup_result.root_address, prefetch_size);
              __atomic_store_n(most_recent_offload_batch_ptr, current_batch_id,
                               __ATOMIC_RELEASE);
            }
          }
        }
      } 
    }
  };
  

  if(prefetch_mode == prefetch_mode::after_sync) {
    int submission_id_in_batch = stdpar::detail::stdpar_tls_runtime::get()
                                   .get_num_outstanding_operations();
    if(submission_id_in_batch == 0)
      (prefetch_handler(args), ...);
  } else if (prefetch_mode == prefetch_mode::always ||
             prefetch_mode == prefetch_mode::first) {
    (prefetch_handler(args), ...);
  } else if (prefetch_mode == prefetch_mode::never) {
    /* nothing to do */
  }
}

template <class AlgorithmType, class Size, typename... Args>
bool should_offload(AlgorithmType type, Size n, const Args &...args) {
  // If we have system USM, no need to validate pointers as all
  // will be automatically valid.
#if !defined(__HIPSYCL_STDPAR_ASSUME_SYSTEM_USM__)
  if(!validate_all_pointers(args...)) {
    HIPSYCL_DEBUG_WARNING << "Detected pointers that are not valid device "
                             "pointers; not offloading stdpar call.\n";
    return false;
  }
#endif

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
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  if (is_offloaded) {                                                          \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
    offload_invoker(q);                                                        \
    hipsycl::stdpar::detail::stdpar_tls_runtime::get()                         \
        .increment_num_outstanding_operations();                               \
  } else {                                                                     \
    fallback_invoker();                                                        \
  }                                                                            \
  __hipsycl_stdpar_optional_barrier(); /*Compiler might move/elide this call*/

#define HIPSYCL_STDPAR_OFFLOAD(algorithm_type_object, problem_size,            \
                               return_type, offload_invoker, fallback_invoker, \
                               ...)                                            \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
  return_type ret = is_offloaded ? offload_invoker(q) : fallback_invoker();    \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::stdpar_tls_runtime::get()                         \
        .increment_num_outstanding_operations();                               \
  __hipsycl_stdpar_optional_barrier(); /*Compiler might move/elide this call*/ \
  return ret;

#define HIPSYCL_STDPAR_BLOCKING_OFFLOAD(algorithm_type_object, problem_size,   \
                                        return_type, offload_invoker,          \
                                        fallback_invoker, ...)                 \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  const auto blocking_fallback_invoker = [&]() {                               \
    q.wait();                                                                  \
    return fallback_invoker();                                                 \
  };                                                                           \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
  return_type ret =                                                            \
      is_offloaded ? offload_invoker(q) : blocking_fallback_invoker();         \
  if (is_offloaded) {                                                          \
    int num_ops = hipsycl::stdpar::detail::stdpar_tls_runtime::get()           \
                      .get_num_outstanding_operations();                       \
    HIPSYCL_DEBUG_INFO                                                         \
        << "[stdpar] Considering " << num_ops                                  \
        << " outstanding operations as completed due to call to "              \
           "blocking stdpar algorithm."                                        \
        << std::endl;                                                          \
    hipsycl::stdpar::detail::stdpar_tls_runtime::get()                         \
        .finalize_offloading_batch();                                          \
  }                                                                            \
  return ret;


} // namespace detail
} // namespace hipsycl::stdpar

#endif
