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

#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#include <cstring>

#include "device_id.hpp"
#include "util.hpp"

namespace hipsycl {
namespace rt {

class backend_executor;
class dag_node;
using dag_node_ptr = std::shared_ptr<dag_node>;

class operation;


namespace hints {

class execution_hint {
public:
  void make_present() {
    _is_present = true;
  }

  bool is_present() const {
    return _is_present;
  }
private:
  bool _is_present = false;
};

class bind_to_device : public execution_hint
{
public:
  bind_to_device() = default;
  explicit bind_to_device(device_id d)
  : _dev{d} {}

  device_id get_device_id() const {
    return _dev;
  }
private:
  device_id _dev;
};


class bind_to_device_group : public execution_hint
{
public:
  bind_to_device_group() = default;
  bind_to_device_group(const std::vector<device_id> &devs)
      : _devs{devs} {
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
  prefer_execution_lane() = default;
  prefer_execution_lane(std::size_t lane_id)
      : _lane_id{lane_id} {}

  std::size_t get_lane_id() const {
    return _lane_id;
  }
private:
  std::size_t _lane_id;
};

class node_group : public execution_hint
{
public:
  node_group() = default;
  node_group(std::size_t group_id)
      : _group_id{group_id} {}

  std::size_t get_id() const {
    return _group_id;
  }
private:
  std::size_t _group_id;
};

class coarse_grained_synchronization : public execution_hint
{};

class prefer_executor : public execution_hint
{
public:
  prefer_executor() = default;
  prefer_executor(std::shared_ptr<backend_executor> executor)
      : _executor{executor.get()}, _shared_executor{executor} {}

  prefer_executor(backend_executor* executor)
      : _executor{executor} {}

  backend_executor* get_executor() const {
    return _executor;
  }
private:
  backend_executor* _executor;
  std::shared_ptr<backend_executor> _shared_executor;
};

class request_instrumentation_submission_timestamp : public execution_hint {};
class request_instrumentation_start_timestamp : public execution_hint {};
class request_instrumentation_finish_timestamp : public execution_hint {};


} // hints


class execution_hints
{
public:
  execution_hints() = default;

  
  template <class HintT,
            std::enable_if_t<std::is_base_of_v<hints::execution_hint, HintT>,
                             int> = 0>
  void set_hint(HintT&& hint) {
    HintT& entry = get_entry<HintT>();
    entry = hint;
    entry.make_present();
  }

  template <class HintT,
            std::enable_if_t<std::is_base_of_v<hints::execution_hint, HintT>,
                             int> = 0>
  const HintT *get_hint() const {
    const HintT& entry = get_entry<HintT>();
    if(entry.is_present())
      return &entry;
    return nullptr;
  }

  template <class HintT> bool has_hint() const {
    const HintT& entry = get_entry<HintT>();
    return entry.is_present();
  }

private:

  template<class T>
  T& get_entry();

  template<class T>
  const T& get_entry() const;


  hints::bind_to_device _bind_to_device;
  hints::bind_to_device_group _bind_to_device_group;
  
  hints::prefer_execution_lane _prefer_execution_lane;
  
  hints::node_group _node_group;
  
  hints::coarse_grained_synchronization _coarse_grained_synchronization;
  
  hints::prefer_executor _prefer_executor;

  hints::request_instrumentation_submission_timestamp
      _request_instrumentation_submission_timestamp;
  hints::request_instrumentation_start_timestamp
      _request_instrumentation_start_timestamp;
  hints::request_instrumentation_finish_timestamp
      _request_instrumentation_finish_timestamp;
};

#define HIPSYCL_RT_HINTS_MAP_GETTER(name, member)                              \
  template <> inline hints::name &execution_hints::get_entry<hints::name>() {  \
    return member;                                                             \
  }                                                                            \
  template <>                                                                  \
  inline const hints::name &execution_hints::get_entry<hints::name>() const {  \
    return member;                                                             \
  }

HIPSYCL_RT_HINTS_MAP_GETTER(bind_to_device, _bind_to_device);
HIPSYCL_RT_HINTS_MAP_GETTER(bind_to_device_group, _bind_to_device_group);
HIPSYCL_RT_HINTS_MAP_GETTER(prefer_execution_lane, _prefer_execution_lane);
HIPSYCL_RT_HINTS_MAP_GETTER(node_group, _node_group);
HIPSYCL_RT_HINTS_MAP_GETTER(coarse_grained_synchronization,
                            _coarse_grained_synchronization);
HIPSYCL_RT_HINTS_MAP_GETTER(prefer_executor, _prefer_executor);
HIPSYCL_RT_HINTS_MAP_GETTER(request_instrumentation_submission_timestamp,
                            _request_instrumentation_submission_timestamp);
HIPSYCL_RT_HINTS_MAP_GETTER(request_instrumentation_start_timestamp,
                            _request_instrumentation_start_timestamp);
HIPSYCL_RT_HINTS_MAP_GETTER(request_instrumentation_finish_timestamp,
                            _request_instrumentation_finish_timestamp);

}
}

#endif
