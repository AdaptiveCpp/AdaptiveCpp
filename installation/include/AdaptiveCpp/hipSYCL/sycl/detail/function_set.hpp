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
#ifndef HIPSYCL_FUNCTION_SET_HPP
#define HIPSYCL_FUNCTION_SET_HPP

#include <unordered_map>
#include "../types.hpp"

namespace hipsycl {
namespace sycl {

class handler;

namespace detail {

template<class Arg>
class function_set
{
public:
  using function_type = function_class<void (Arg)>;
  using id = std::size_t;

  void run_all(Arg arg) const
  {
    for(const auto& element : _functions)
      element.second(arg);
  }

  id add(function_type&& f)
  {
    id new_id = static_cast<id>(_functions.size());
    _functions[new_id] = f;
    return new_id;
  }

  void remove(id function_id)
  {
    _functions.erase(function_id);
  }

private:
  using function_map_type = 
    std::unordered_map<id, function_type>;
  
  function_map_type _functions;
};


}
}
}


#endif
