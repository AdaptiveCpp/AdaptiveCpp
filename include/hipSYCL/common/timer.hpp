/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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


#include <chrono>
#include <vector>
#include <string>
#include "debug.hpp"

namespace hipsycl::common {

class timer {
public:
  timer(const std::string& name, bool print_at_destruction = false)
  : _name{name}, _print{print_at_destruction} {
    _start = std::chrono::high_resolution_clock::now();
    _is_running = true;
  }

  double stop() {
    if(_is_running) {
      _stop = std::chrono::high_resolution_clock::now();
      _is_running = false;
    }

    auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(_stop - _start).count();
    return static_cast<double>(ticks) * 1.e-9;
  }

  double stop_and_print() {
    double T = stop();
    HIPSYCL_DEBUG_INFO << "Phase '" << _name << "' took " << T << " seconds\n"; 
    return T;
  }

  ~timer() {
    if(_print)
      stop_and_print();
    else
      stop();
  }
private:
  bool _print;
  bool _is_running = false;
  std::string _name;

  using time_point_t = 
    std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_t _start;
  time_point_t _stop;
  
};

}