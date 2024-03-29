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
#include "hipSYCL/runtime/settings.hpp"
#include "hipSYCL/std/stdpar/detail/execution_fwd.hpp"
#include "hipSYCL/std/stdpar/detail/stdpar_builtins.hpp"
#include "hipSYCL/std/stdpar/detail/sycl_glue.hpp"
#include "hipSYCL/std/stdpar/detail/offload_heuristic_db.hpp"

#include "hipSYCL/glue/reflection.hpp"
#include "hipSYCL/common/stable_running_hash.hpp"


#include <atomic>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <cstddef>
#include <algorithm>
#include <chrono>
#include <limits>
#include <sys/types.h>
#include <utility>

namespace hipsycl::stdpar {

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

template<class Handler, typename... Args>
void for_each_contained_pointer(Handler&& h, const Args&... args) {
  auto f = [&](const auto& arg){
    if(!has_decoration<decorations::no_pointer_validation>(arg)) {
      glue::reflection::introspect_flattened_struct introspection{arg};
      for(int i = 0; i < introspection.get_num_members(); ++i) {
        if(introspection.get_member_kind(i) == glue::reflection::type_kind::pointer) {
          void* ptr = nullptr;
          std::memcpy(&ptr, (char*)&arg + introspection.get_member_offset(i), sizeof(void*));
          // Ignore nullptr
          if(ptr) {
            h(ptr);
          }
        }
      }
    };
  };
  (f(args), ...);
}

template<typename... Args>
bool validate_all_pointers(const Args&... args){
  bool result = true;
  
  auto& q = detail::single_device_dispatch::get_queue();
  auto* allocator = q.get_context()
      .hipSYCL_runtime()
      ->backends()
      .get(q.get_device().get_backend())
      ->get_allocator(q.get_device().hipSYCL_device_id());

  auto f = [&](const void* ptr){
    if(ptr) {
      rt::pointer_info pinfo;
      if(!allocator->query_pointer(ptr, pinfo).is_success())
        result = false;
    }
  };

  for_each_contained_pointer(f, args...);

  return result;
}

enum prefetch_mode {
  automatic = 0,
  always = 1,
  never = 2,
  after_sync = 3,
  first = 4
};

inline prefetch_mode get_prefetch_mode() noexcept {
#ifdef __HIPSYCL_STDPAR_PREFETCH_MODE__
  prefetch_mode mode = static_cast<prefetch_mode>(__HIPSYCL_STDPAR_PREFETCH_MODE__);
#else
  auto determine_prefetch_mode = [&]() -> prefetch_mode {
    std::string prefetch_mode_string;
    if(rt::try_get_environment_variable("stdpar_prefetch_mode", prefetch_mode_string)) {
      if(prefetch_mode_string == "auto") {
        return prefetch_mode::automatic;
      } else if(prefetch_mode_string == "always") {
        return prefetch_mode::always;
      } else if(prefetch_mode_string == "never") {
        return prefetch_mode::never;
      } else if(prefetch_mode_string == "always") {
        return prefetch_mode::always;
      } else if(prefetch_mode_string == "after-sync") {
        return prefetch_mode::after_sync;
      } else if(prefetch_mode_string == "first") {
        return prefetch_mode::first;
      } else {
        HIPSYCL_DEBUG_ERROR << "Invalid prefetch mode: " << prefetch_mode_string
                            << ", falling back to 'auto'\n";
      }
    }
    return prefetch_mode::automatic;
  };

  static prefetch_mode mode = determine_prefetch_mode();
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

#ifndef __HIPSYCL_STDPAR_ASSUME_SYSTEM_USM__
  // Use "first" mode in case of automatic prefetch decision for now
  const auto prefetch_mode =
      (get_prefetch_mode() == prefetch_mode::automatic) ? prefetch_mode::first
                                                        : get_prefetch_mode();

  auto prefetch_handler = [&](void* ptr){
    unified_shared_memory::allocation_lookup_result lookup_result;
    
    if(ptr && unified_shared_memory::allocation_lookup(ptr, lookup_result)) {
      int64_t *most_recent_offload_batch_ptr =
          &(lookup_result.info->most_recent_offload_batch);

      std::size_t prefetch_size = lookup_result.info->allocation_size;

      // Need to use atomic builtins until we can use C++ 20 atomic_ref :(
      int64_t most_recent_offload_batch = __atomic_load_n(
          most_recent_offload_batch_ptr, __ATOMIC_ACQUIRE);
      
      bool should_prefetch = false;
      if(prefetch_mode == prefetch_mode::first)
        // an allocation that was never used will still contain the
        // initialization value of -1
        should_prefetch = most_recent_offload_batch == -1;
      else
        // Never emit multiple prefetches for the same allocation in one batch
        should_prefetch = most_recent_offload_batch <
                          static_cast<int64_t>(current_batch_id);

      if (should_prefetch) {
        //sycl::mem_advise(lookup_result.root_address, prefetch_size, 3, q);
        prefetch(q, lookup_result.root_address, prefetch_size);
        __atomic_store_n(most_recent_offload_batch_ptr, current_batch_id,
                          __ATOMIC_RELEASE);
      }
    }
  };
  

  if(prefetch_mode == prefetch_mode::after_sync) {
    int submission_id_in_batch = stdpar::detail::stdpar_tls_runtime::get()
                                   .get_num_outstanding_operations();
    if(submission_id_in_batch == 0)
      for_each_contained_pointer(prefetch_handler, args...);
  } else if (prefetch_mode == prefetch_mode::always ||
             prefetch_mode == prefetch_mode::first) {
    for_each_contained_pointer(prefetch_handler, args...);
  } else if (prefetch_mode == prefetch_mode::never) {
    /* nothing to do */
  }
#endif
}

struct pair_hash{
  template <class T1, class T2>
  std::size_t operator() (const std::pair<T1, T2> &pair) const {
    return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
  }
};
template <class K, class V>
using host_malloc_unordered_pair_map =
    std::unordered_map<K, V, pair_hash, std::equal_to<K>,
                       libc_allocator<std::pair<const K, V>>>;

class offload_heuristic_config {
public:
  offload_heuristic_config() {
    if(!rt::try_get_environment_variable("stdpar_ohc_min_ops", _min_ops_per_offload_decision)) {
      _min_ops_per_offload_decision = 128;
    }
    if(!rt::try_get_environment_variable("stdpar_ohc_min_time", _min_time_per_offload_decision)) {
      _min_time_per_offload_decision = 1;
    }
    // Convert from seconds to ns
    _min_time_per_offload_decision *= 1.e9;
  }

  double get_min_time_per_offload_decision() const {
    return _min_time_per_offload_decision;
  }

  int get_min_ops_per_offload_decision() const {
    return _min_time_per_offload_decision;
  }
private:
  double _min_time_per_offload_decision;
  int _min_ops_per_offload_decision;
};

struct offload_heuristic_state {
  static offload_heuristic_state& get() {
    static thread_local offload_heuristic_state state;
    return state;
  }

  // op_id is pair of hash and problem size
  using op_id = std::pair<uint64_t, std::size_t>;

  void proceed_to(uint64_t op_hash, std::size_t problem_size) {

    _most_recent_successor[_previous_op] = {op_hash, problem_size};
    _previous_op = {op_hash, problem_size};
    
    uint64_t now = get_time_now();
    _time_since_previous_op = now - _previous_offloading_change_timestamp;

    ++_num_ops_since_offloading_change;
    ++_num_total_ops;
  }

  std::optional<op_id> predict_next(op_id op) const {
    auto it = _most_recent_successor.find(op);
    if(it == _most_recent_successor.end())
      return {};
    return it->second;
  }

  void set_offloading(bool offloading) {
    _is_currently_offloading = offloading;

    _num_ops_since_offloading_change = 0;
    _previous_offloading_change_timestamp = get_time_now();
  }

  bool is_currently_offloading() const {
    return _is_currently_offloading;
  }

  uint64_t get_ns_since_previous_op() const {
    return _time_since_previous_op;
  }

  std::size_t get_num_total_ops() const {
    return _num_total_ops;
  }

  std::size_t get_num_ops_since_offloading_change() const {
    return _num_ops_since_offloading_change;
  }

  bool is_host_sampling_run() const {
    return _is_host_sampling_run;
  }

  bool is_offload_sampling_run() const {
    return _is_offload_sampling_run;
  }

  void set_num_predicted_ops(int n) {
    _num_predicted_ops = n;
  }

  int get_num_predicted_ops() const {
    return _num_predicted_ops;
  }
  
  const offload_heuristic_config& get_configuration() const {
    return _config;
  }
private:
  
  offload_heuristic_state()
      : _is_host_sampling_run{is_host_sampling_run_requested()},
        _is_offload_sampling_run{is_offload_sampling_run_requested()},
        _num_ops_since_offloading_change{0}, _previous_op{},
        _previous_offloading_change_timestamp{0}, _time_since_previous_op{0},
        _is_currently_offloading{false}, _num_total_ops{0}, _num_predicted_ops{0} {}

  bool _is_host_sampling_run;
  bool _is_offload_sampling_run;
  int _num_ops_since_offloading_change;
  op_id _previous_op;
  host_malloc_unordered_pair_map<op_id, op_id> _most_recent_successor;
  uint64_t _previous_offloading_change_timestamp;
  uint64_t _time_since_previous_op;
  bool _is_currently_offloading;
  std::size_t _num_total_ops;
  int _num_predicted_ops;
  offload_heuristic_config _config;

  static bool is_host_sampling_run_requested(){
    bool is_requested = false;
    if (rt::try_get_environment_variable("stdpar_host_sampling",
                                          is_requested))
      return is_requested;
    return false;
  }

  static bool is_offload_sampling_run_requested(){
    bool is_requested = false;
    if (rt::try_get_environment_variable("stdpar_offload_sampling",
                                          is_requested))
      return is_requested;
    return false;
  }
};

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


  offload_heuristic_state& state = offload_heuristic_state::get();
  if(state.is_host_sampling_run())
    return false;
  else if(state.is_offload_sampling_run())
    return true;

  int min_ops_before_offloading_change =
      state.get_configuration().get_min_ops_per_offload_decision();

  int num_predicted_ops = 0;
  uint64_t op_hash = get_operation_hash(type, n, args...);
  // Identify ops using a combination of hash and problem size
  using op_id = offload_heuristic_state::op_id;

  auto for_each_known_op_in_batch = [&](auto handler) {
    int max_iters = min_ops_before_offloading_change;
    op_id current = {op_hash, n};
    for(int num_iters = 0; num_iters < max_iters; ++num_iters) {
      
      if(!handler(current))
        return;

      auto prediction = state.predict_next(current);
      
      if(!prediction.has_value())
        return;
      current = prediction.value();
    }
  };

  auto decide_offloading_viability = [&](std::optional<bool> is_currently_offloading = {}){

    // Instead of hardcoding peak PCIe speeds, this should be measured
    double data_transfer_time_estimate = 0;

#if !defined(__HIPSYCL_STDPAR_ASSUME_SYSTEM_USM__)
    std::size_t used_memory = 0;
    for_each_contained_pointer([&](void* ptr){
      unified_shared_memory::allocation_lookup_result lookup_result;
  
      if(ptr && unified_shared_memory::allocation_lookup(ptr, lookup_result)) {
        used_memory += lookup_result.info->allocation_size;
      }
    }, args...);

    if(detail::stdpar_tls_runtime::get().get_current_offloading_batch_id() > 0)
      data_transfer_time_estimate = used_memory / 32.0;
#endif

    double host_time_estimate = 0.0;
    double offload_time_estimate = 0.0;
    
    num_predicted_ops = 0;

    for_each_known_op_in_batch([&](op_id op) -> bool{
      auto& db = detail::stdpar_tls_runtime::get().get_offload_db();
      double current_host_estimate = db.estimate_runtime(
          op.first, op.second, offload_heuristic_db::host_device_id);
      double current_offload_estimate = db.estimate_runtime(
          op.first, op.second, offload_heuristic_db::offload_device_id);
      
      if(current_host_estimate <= 0.0 || current_offload_estimate <= 0.0) {
        // Abort when we have no data for a given operation
        return false;
      }

      host_time_estimate += current_host_estimate;
      offload_time_estimate += current_offload_estimate;
      ++num_predicted_ops;

      return true;
    });

    

    if(host_time_estimate <= 0.0)
      // If we don't have host sampling data, offload.
      return true;
    
    if(is_currently_offloading.has_value()){
      if(is_currently_offloading.value()) {
        host_time_estimate += data_transfer_time_estimate;
      } else {
        offload_time_estimate += data_transfer_time_estimate;
      }

      double ratio = host_time_estimate / offload_time_estimate;
      double tolerance = 0.2;
      if(ratio >= (1.0 - tolerance) && ratio <= (1.0 + tolerance))
        return is_currently_offloading.value();

    }
    
    return offload_time_estimate < host_time_estimate;
    
  };

  if(state.get_num_total_ops() == 0) {
    state.set_offloading(decide_offloading_viability());
  }

  state.proceed_to(op_hash, n);
  
  if (state.get_num_ops_since_offloading_change() < state.get_num_predicted_ops() ||
      state.get_num_ops_since_offloading_change() < min_ops_before_offloading_change ||
      state.get_ns_since_previous_op() < state.get_configuration().get_min_time_per_offload_decision()) {
    return state.is_currently_offloading();
  } else {
    state.set_offloading(decide_offloading_viability(state.is_currently_offloading()));
    state.set_num_predicted_ops(num_predicted_ops);

    HIPSYCL_DEBUG_INFO << "[stdpar] Offloading behavior decision: "
                       << (state.is_currently_offloading() ? "Offloading!"
                                                           : "Offloading disabled.")
                       << "\n";
  }

  return state.is_currently_offloading();
#endif
}

struct host_invocation_measurement {
  host_invocation_measurement(uint64_t hash, std::size_t problem_size)
  : _hash{hash}, _problem_size{problem_size} {}

  template<class F>
  auto operator()(F&& f) {

    auto& offload_db = stdpar_tls_runtime::get().get_offload_db();
    
    _start = get_time_now();

    return f();
  }

  ~host_invocation_measurement() {
    auto& offload_db = stdpar_tls_runtime::get().get_offload_db();
    
      uint64_t end = get_time_now();
      uint64_t delta = end - _start;

      offload_db.update_entry(_hash, _problem_size,
                              offload_heuristic_db::host_device_id,
                              (double)delta);
  }
private:
  uint64_t _hash;
  std::size_t _problem_size;
  uint64_t _start = 0;
};

struct device_invocation_measurement {
  device_invocation_measurement(uint64_t hash, std::size_t problem_size)
  : _hash{hash}, _problem_size{problem_size} {}

  template<class F>
  auto operator()(F&& f) {
 
    auto& offload_db = stdpar_tls_runtime::get().get_offload_db();
    stdpar_tls_runtime::get().instrument_offloaded_operation(_hash, _problem_size);

    return f();
  }
private:
  uint64_t _hash;
  std::size_t _problem_size;
};

template<class AlgorithmType, class Size, class F, typename... Args>
auto host_instrumentation(F&& f, AlgorithmType t, Size n, Args... args) {
#ifndef __HIPSYCL_STDPAR_UNCONDITIONAL_OFFLOAD__
  uint64_t hash = get_operation_hash(t, n, args...);
  host_invocation_measurement m{hash, n};
  return m(f);
#else
  return f();
#endif
}

template<class AlgorithmType, class Size, class F, typename... Args>
auto device_instrumentation(F&& f, AlgorithmType t, Size n, Args... args) {
#ifndef __HIPSYCL_STDPAR_UNCONDITIONAL_OFFLOAD__
  uint64_t hash = get_operation_hash(t, n, args...);
  device_invocation_measurement m{hash, n};
  return m(f);
#else
  return f();
#endif
}

#define HIPSYCL_STDPAR_OFFLOAD_NORET(algorithm_type_object, problem_size,      \
                                     offload_invoker, fallback_invoker, ...)   \
  using hipsycl::stdpar::detail::device_instrumentation;                       \
  using hipsycl::stdpar::detail::host_instrumentation;                         \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  if (is_offloaded) {                                                          \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
                                                                               \
    device_instrumentation([&]() { offload_invoker(q); },                      \
                           algorithm_type_object, problem_size, __VA_ARGS__);  \
    hipsycl::stdpar::detail::stdpar_tls_runtime::get()                         \
        .increment_num_outstanding_operations();                               \
  } else {                                                                     \
    __hipsycl_stdpar_barrier();                                                \
    host_instrumentation([&]() { fallback_invoker(); }, algorithm_type_object, \
                         problem_size, __VA_ARGS__);                           \
  }                                                                            \
  __hipsycl_stdpar_optional_barrier(); /*Compiler might move/elide this call*/

#define HIPSYCL_STDPAR_OFFLOAD(algorithm_type_object, problem_size,            \
                               return_type, offload_invoker, fallback_invoker, \
                               ...)                                            \
  using hipsycl::stdpar::detail::device_instrumentation;                       \
  using hipsycl::stdpar::detail::host_instrumentation;                         \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
  else                                                                         \
    __hipsycl_stdpar_barrier();                                                \
  return_type ret =                                                            \
      is_offloaded                                                             \
          ? device_instrumentation([&]() { return offload_invoker(q); },       \
                                   algorithm_type_object, problem_size,        \
                                   __VA_ARGS__)                                \
          : host_instrumentation([&]() { return fallback_invoker(); },         \
                                 algorithm_type_object, problem_size,          \
                                 __VA_ARGS__);                                 \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::stdpar_tls_runtime::get()                         \
        .increment_num_outstanding_operations();                               \
  __hipsycl_stdpar_optional_barrier(); /*Compiler might move/elide this call*/ \
  return ret;

#define HIPSYCL_STDPAR_BLOCKING_OFFLOAD(algorithm_type_object, problem_size,   \
                                        return_type, offload_invoker,          \
                                        fallback_invoker, ...)                 \
  using hipsycl::stdpar::detail::device_instrumentation;                       \
  using hipsycl::stdpar::detail::host_instrumentation;                         \
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();      \
  bool is_offloaded = hipsycl::stdpar::detail::should_offload(                 \
      algorithm_type_object, problem_size, __VA_ARGS__);                       \
  const auto blocking_fallback_invoker = [&]() {                               \
    q.wait();                                                                  \
    return host_instrumentation([&]() { return fallback_invoker(); },          \
                                algorithm_type_object, problem_size,           \
                                __VA_ARGS__);                                  \
  };                                                                           \
  if (is_offloaded)                                                            \
    hipsycl::stdpar::detail::prepare_offloading(algorithm_type_object,         \
                                                problem_size, __VA_ARGS__);    \
  else                                                                         \
    __hipsycl_stdpar_barrier();                                                \
  return_type ret =                                                            \
      is_offloaded                                                             \
          ? device_instrumentation([&]() { return offload_invoker(q); },       \
                                   algorithm_type_object, problem_size,        \
                                   __VA_ARGS__)                                \
          : blocking_fallback_invoker();                                       \
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
