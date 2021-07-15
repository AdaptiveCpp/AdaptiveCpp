/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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
  /// Wait for the given instrumentation to make results available. 
  /// Returns nullptr if the given instrumentation was not set up.
  template<typename Instr> const std::shared_ptr<Instr> get() const {
    // First wait until all instrumentations have been set up
    if(!_registration_complete_signal.has_signalled()){
      _registration_complete_signal.wait();
    }
    
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
    assert(!_registration_complete_signal.has_signalled());
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
    _registration_complete_signal.signal();
  }

private:
  std::vector<std::pair<std::type_index, std::shared_ptr<instrumentation>>>
      _instrs;

  mutable signal_channel _registration_complete_signal;
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
