/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_INSTRUMENTATION_HPP
#define HIPSYCL_INSTRUMENTATION_HPP

#include <cassert>
#include <cstdint>
#include <chrono>
#include <future>
#include <vector>
#include <typeindex>
#include <type_traits>

#include "signal_channel.hpp"

namespace hipsycl {
namespace rt {

// SYCL defines timestamps as uint64_t with nanosecond resolution
class profiler_clock {
public:
  using rep = uint64_t;
  using period = std::nano;
  using duration = std::chrono::duration<rep, period>;
  using time_point = std::chrono::time_point<profiler_clock>;

  constexpr static bool is_steady = true;

  static time_point now() {
    return time_point{
        duration{std::chrono::steady_clock::now().time_since_epoch()}};
  }

  static std::size_t ns_ticks(const time_point& tp) {
    return tp.time_since_epoch().count();
  }

  static double seconds(const time_point& tp) {
    auto ticks = ns_ticks(tp);
    return static_cast<double>(ticks) /
           1.e9;
  }
};

class instrumentation {
public:
  /// Waits until an instrumentation has its result available
  /// This does not need to be called manually, as the instrumentation_set
  /// will call it automatically if an instrumentation result is requested.
  virtual void wait() const = 0;
  virtual ~instrumentation() = default;
};

template<typename T>
inline constexpr bool is_instrumentation_v
    = std::is_convertible_v<std::remove_cv_t<T> *, instrumentation *>;

class instrumentation_set {
public:
  instrumentation_set()
  : _registration_complete{false} {}

  /// Wait for the given instrumentation to make results available. 
  /// Returns nullptr if the given instrumentation was not set up.
  template<typename Instr> const std::shared_ptr<Instr> get() const {
    // First wait until all instrumentations have been set up
    
    while(!_registration_complete)
      ;
    
    std::shared_ptr<Instr> i = nullptr;

    for(const auto& current : _instrs) {
      if(current.first == typeid(Instr)) {
        i = std::static_pointer_cast<Instr>(current.second);
      }
    }

    if(!i)
      return nullptr;
    // Wait for the target instrumentation to return results
    i->wait();

    return i;
  }

  template<typename Instr>
  void add_instrumentation(std::shared_ptr<Instr> instr) {
    assert(!_registration_complete);
    for(const auto& i : _instrs) {
      if(i.first == typeid(Instr)) {
        assert(false && "Instrumentation already exists!");
      }
    }
    std::type_index idx = typeid(Instr);
    _instrs.push_back(std::make_pair(idx, instr));
  }

  // This will be called by the scheduler after node submission.
  // After calling, no additional instrumentations can be added anymore.
  void mark_set_complete() {
    // This used to rely on signal_channel.signal(), however
    // in fast submission loops this can add overhead even if no instrumentations
    // are used. We are now using a spin-lock which has its own issues, but
    // at least the cost is very low when not using instrumentations.
    _registration_complete = true;
  }

private:
  std::vector<std::pair<std::type_index, std::shared_ptr<instrumentation>>>
      _instrs;

  // TODO Change to atomic_flag so we can use wait() once we have C++20
  std::atomic<bool> _registration_complete;
};

namespace instrumentations {

class submission_timestamp : public instrumentation {
public:
  virtual profiler_clock::time_point get_time_point() const = 0;
  virtual ~submission_timestamp() = default;
};

class execution_start_timestamp : public instrumentation {
public:
  virtual profiler_clock::time_point get_time_point() const = 0;
  virtual ~execution_start_timestamp() = default;
};

class execution_finish_timestamp : public instrumentation {
public:
  virtual profiler_clock::time_point get_time_point() const = 0;
  virtual ~execution_finish_timestamp() = default;
};

}

class simple_submission_timestamp : public instrumentations::submission_timestamp {
public:
  simple_submission_timestamp(profiler_clock::time_point submission_time)
  : _time{submission_time} {}

  virtual profiler_clock::time_point get_time_point() const override {
    return _time;
  }

  virtual void wait() const override {}
private:
  profiler_clock::time_point _time;
};

}
}

#endif
