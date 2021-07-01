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

#include <cstdint>
#include <chrono>
#include <future>
#include <unordered_map>
#include <typeindex>
#include <type_traits>

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
    return time_point{duration{std::chrono::steady_clock::now().time_since_epoch()}};
  }
};

class instrumentation {
public:
  virtual ~instrumentation() = default;
};

template<typename T>
inline constexpr bool is_instrumentation_v
    = std::is_convertible_v<std::remove_cv_t<T> *, instrumentation *>;

// Provides synchronization points for instrumentations between front- and backend threads.
//
// Thread safety: Before submission - single-threaded access only
//                After submission: - concurrent access, but do not call instrument<>()
class instrumentation_set {
public:
  // Create a producer-consumer channel for a particular instrumentation
  template<typename Instr>
  void instrument() {
    static_assert(is_instrumentation_v<Instr>);
    if (_instrs.find(typeid(Instr)) == _instrs.end()) {
      _instrs.emplace(typeid(Instr), future_instrumentation{});
    }
  }

  // Whether instrument<Instr>() has been called before.
  template<typename Instr>
  bool is_instrumented() const {
    static_assert(is_instrumentation_v<Instr>);
    return _instrs.find(typeid(Instr)) != _instrs.end();
  }

  // Resolve the instrument<Instr>() promise by providing an instrumentation instance.
  // instrument<Instr>() must have been called before submission.
  template<typename Instr, typename T>
  void provide(std::unique_ptr<T> instr) {
    static_assert(is_instrumentation_v<Instr>);
    static_assert(std::is_convertible_v<T*, Instr*>);
    _instrs.at(typeid(Instr)).provide(std::move(instr));
  }

  // Wait until another thread provide()s an instance of Instr.
  // instrument<Instr>() must have been called before submission.
  template<typename Instr> const Instr &await() const {
    static_assert(is_instrumentation_v<Instr>);
      // static_cast: type is statically checked in resolve()
    return static_cast<const Instr &>(_instrs.at(typeid(Instr)).await());
  }

private:
  class future_instrumentation {
  public:
    future_instrumentation(): _future{_promise.get_future()} {}
    void provide(std::unique_ptr<instrumentation> instr) { _promise.set_value(std::move(instr)); }
    const instrumentation &await() const { return *_future.get(); }

  private:
    std::promise<std::unique_ptr<instrumentation>> _promise;
    std::shared_future<std::unique_ptr<instrumentation>> _future;
  };

  std::unordered_map<std::type_index, future_instrumentation> _instrs;
};

class timestamp_profiler: public instrumentation
{
public:
  enum class event
  {
    operation_submitted,
    operation_started,
    operation_finished
  };

  virtual profiler_clock::time_point await_event(event) const = 0;
};

}
}

#endif
