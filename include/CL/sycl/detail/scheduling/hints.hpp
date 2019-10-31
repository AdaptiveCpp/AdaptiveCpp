/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_HINTS_HPP
#define HIPSYCL_HINTS_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include "device_id.hpp"

namespace cl {
namespace sycl { 
namespace detail {


class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

enum class execution_hint_type
{
  use_device,
  use_pseudo_queue_id,
  explicit_require
};

class execution_hint
{
public:
  execution_hint(execution_hint_type type)
  : _type{type}
  {}

  execution_hint_type get_hint_type() const
  {
    return _type;
  }

  virtual ~execution_hint(){}
private:
  execution_hint_type _type;
};

using execution_hint_ptr = std::shared_ptr<execution_hint>;

template<class T, typename... Args>
execution_hint_ptr make_execution_hint(Args... args)
{
  return execution_hint_ptr{
    new T{args...}
  };
}

namespace hints {

class use_device : public execution_hint
{
public:
  use_device(device_id d)
  : _dev{d}, execution_hint{execution_hint_type::use_device}
  {}

  device_id get_device_id() const
  { return _dev; }
private:
  device_id _dev;
};

class explicit_require : public execution_hint
{
public:
  explicit_require(dag_node_ptr node)
  : _dag_node{node}, 
    execution_hint{execution_hint_type::explicit_require}
  {}

  dag_node_ptr get_requirement() const
  {
    return _dag_node;
  }

private:
  dag_node_ptr _dag_node;
};

}

class execution_hints
{
public:
  void add_hint(execution_hint_ptr hint)
  {
    _hints.push_back(hint);
  }

  void overwrite_with(const execution_hints& other)
  {
    for(const auto& hint : other._hints)
    {
      execution_hint_type type = hint->get_hint_type();
      auto it = std::find_if(_hints.begin(),_hints.end(),
        [type](execution_hint_ptr h){
        return type == h->get_hint_type();
      });

      if(it != _hints.end())
        *it = hint;
    }
  }

  bool has_hint(execution_hint_type type) const
  {
    return get_hint(type) != nullptr;
  }

  execution_hint_ptr get_hint(execution_hint_type type) const
  {
    for(const auto& hint : _hints)
      if(hint->get_hint_type() == type)
        return hint;
    return nullptr;
  }

  template<execution_hint_type Type>
  execution_hint_ptr get_hint() const
  {
    return get_hint(Type);
  }

private:
  std::vector<execution_hint_ptr> _hints;
};

}
}
}

#endif
