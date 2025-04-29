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
#ifndef HIPSYCL_ASYNC_ERRORS_HPP
#define HIPSYCL_ASYNC_ERRORS_HPP

#include <mutex>
#include <vector>

#include "error.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/small_vector.hpp"

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
  common::auto_small_vector<result> _errors;
};

}  
}

#endif