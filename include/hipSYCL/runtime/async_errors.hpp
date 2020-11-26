/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HIPSYCL_ASYNC_ERRORS_HPP
#define HIPSYCL_ASYNC_ERRORS_HPP

#include <mutex>
#include <vector>

#include "error.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

class async_error_list
{
public:

  void add(const result& res)
  {
    std::lock_guard<std::mutex> lock{_lock};

    print_result(res);
    
    _errors.push_back(res);
  }

  void clear() {
    std::lock_guard<std::mutex> lock{_lock};

    _errors.clear();
  }

  template<class F>
  void for_each_error(F handler) {
    std::lock_guard<std::mutex> lock{_lock};
    for(const auto& err : _errors)
      handler(err);
  }


  template<class F>
  void pop_each_error(F handler) {
    std::lock_guard<std::mutex> lock{_lock};
    for(const auto& err : _errors)
      handler(err);
    _errors.clear();
  }

  std::size_t num_errors() const {
    std::lock_guard<std::mutex> lock{_lock};
    return _errors.size();
  }
private:
  mutable std::mutex _lock;
  std::vector<result> _errors;
};

}  
}

#endif