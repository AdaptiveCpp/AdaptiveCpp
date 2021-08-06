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

#ifndef HIPSYCL_HINTS_HPP
#define HIPSYCL_HINTS_HPP

#include <algorithm>
#include <memory>
#include <vector>

#include "device_id.hpp"
#include "util.hpp"

namespace hipsycl {
namespace rt {


class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

class operation;

enum class execution_hint_type
{
  // mark a DAG node as bound to a particular device for execution
  bind_to_device,
  bind_to_device_group,
  prefer_execution_lane,
  node_group,

  request_instrumentation_submission_timestamp,
  request_instrumentation_start_timestamp,
  request_instrumentation_finish_timestamp
};

class execution_hint
{
public:
  execution_hint(execution_hint_type type);

  execution_hint_type get_hint_type() const;

  virtual ~execution_hint();
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

class bind_to_device : public execution_hint
{
public:
  static constexpr execution_hint_type type = 
    execution_hint_type::bind_to_device;

  explicit bind_to_device(device_id d);

  device_id get_device_id() const;
private:
  device_id _dev;
};


class bind_to_device_group : public execution_hint
{
public:
  static constexpr execution_hint_type type = 
    execution_hint_type::bind_to_device_group;

  bind_to_device_group(const std::vector<device_id> &devs)
      : execution_hint{execution_hint_type::bind_to_device_group}, _devs{devs} {
  }

  const std::vector<device_id>& get_devices() const {
    return _devs;
  }
private:
  std::vector<device_id> _devs;
};


class prefer_execution_lane : public execution_hint
{
public:
  static constexpr execution_hint_type type =
      execution_hint_type::prefer_execution_lane;

  prefer_execution_lane(std::size_t lane_id)
      : execution_hint{execution_hint_type::prefer_execution_lane},
        _lane_id{lane_id} {}

  std::size_t get_lane_id() const {
    return _lane_id;
  }
private:
  std::size_t _lane_id;
};

class node_group : public execution_hint
{
public:
  static constexpr execution_hint_type type =
      execution_hint_type::node_group;

  node_group(std::size_t group_id)
      : execution_hint{execution_hint_type::node_group}, _group_id{group_id} {}

  std::size_t get_id() const {
    return _group_id;
  }
private:
  std::size_t _group_id;
};

class request_instrumentation_submission_timestamp : public execution_hint {
public:
  static constexpr execution_hint_type type =
      execution_hint_type::request_instrumentation_submission_timestamp;

  request_instrumentation_submission_timestamp()
      : execution_hint{execution_hint_type::
                           request_instrumentation_submission_timestamp} {}
};


class request_instrumentation_start_timestamp : public execution_hint {
public:
  static constexpr execution_hint_type type =
      execution_hint_type::request_instrumentation_start_timestamp;

  request_instrumentation_start_timestamp()
      : execution_hint{execution_hint_type::
                           request_instrumentation_start_timestamp} {}
};

class request_instrumentation_finish_timestamp : public execution_hint {
public:
  static constexpr execution_hint_type type =
      execution_hint_type::request_instrumentation_finish_timestamp;

  request_instrumentation_finish_timestamp()
      : execution_hint{execution_hint_type::
                           request_instrumentation_finish_timestamp} {}
};

} // hints



class execution_hints
{
public:
  void add_hint(execution_hint_ptr hint);
  void overwrite_with(const execution_hints &other);
  void overwrite_with(execution_hint_ptr hint);
  
  bool has_hint(execution_hint_type type) const;
  execution_hint* get_hint(execution_hint_type type) const;

  template<class Hint_type>
  Hint_type* get_hint() const
  {
    execution_hint* ptr = get_hint(Hint_type::type);
    if(ptr)
      return cast<Hint_type>(ptr);
    return nullptr;
  }

  template <class Hint_type> bool has_hint() const {
    return get_hint(Hint_type::type) != nullptr;
  }

  friend bool operator==(const execution_hints &a, const execution_hints &b) {
    return a._hints == b._hints;
  }

  friend bool operator!=(const execution_hints &a, const execution_hints &b) {
    return !(a==b);
  }
private:
  std::vector<execution_hint_ptr> _hints;
};


}
}

#endif
