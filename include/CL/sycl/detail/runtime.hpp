/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef HIPSYCL_RUNTIME_HPP
#define HIPSYCL_RUNTIME_HPP

#include <unordered_map>
#include <cassert>

#include "task_graph.hpp"
#include "buffer.hpp"
#include "../access.hpp"
#include "../types.hpp"

namespace cl {
namespace sycl {
namespace detail {

template<typename dataT, int dimensions,
         access::mode accessmode,
         access::target accessTarget,
         access::placeholder isPlaceholder>
class accessor_base;

class placeholder_accessor_tracker
{
public:

  template<typename dataT,
           int dimensions,
           access::mode accessmode,
           access::target accessTarget>
  using placeholder_accessor =
      accessor_base<dataT,dimensions,accessmode,accessTarget,access::placeholder::true_t>;

  template<typename dataT, int dimensions,
           access::mode accessmode,
           access::target accessTarget>
  void new_accessor(
      const placeholder_accessor<dataT,dimensions,accessmode,accessTarget>* accessor_ptr,
      buffer_ptr buff)
  {
    std::lock_guard<mutex_class> lock{_lock};

    const void* placeholder_object_ptr = reinterpret_cast<const void*>(accessor_ptr);
    assert(_placeholder_buffer_map.find(placeholder_object_ptr) ==
           _placeholder_buffer_map.end());

    _placeholder_buffer_map[placeholder_object_ptr] = buff;
  }

  template<typename dataT, int dimensions,
           access::mode accessmode,
           access::target accessTarget>
  void set_accessor_buffer(
      const placeholder_accessor<dataT,dimensions,accessmode,accessTarget>* accessor_ptr,
      buffer_ptr buff)
  {
    std::lock_guard<mutex_class> lock{_lock};

    const void* placeholder_object_ptr = reinterpret_cast<const void*>(accessor_ptr);
    assert(_placeholder_buffer_map.find(placeholder_object_ptr) !=
           _placeholder_buffer_map.end());

    _placeholder_buffer_map[placeholder_object_ptr] = buff;
  }

  template<typename dataT, int dimensions,
           access::mode accessmode,
           access::target accessTarget>
  void release_accessor(
      const placeholder_accessor<dataT,dimensions,accessmode,accessTarget>* accessor_ptr)
  {
    std::lock_guard<mutex_class> lock{_lock};

    const void* placeholder_object_ptr = reinterpret_cast<const void*>(accessor_ptr);
    assert(_placeholder_buffer_map.find(placeholder_object_ptr) !=
           _placeholder_buffer_map.end());

    _placeholder_buffer_map.erase(placeholder_object_ptr);
  }

  template<typename dataT, int dimensions,
           access::mode accessmode,
           access::target accessTarget>
  buffer_ptr find_accessor(
      const placeholder_accessor<dataT,dimensions,accessmode,accessTarget>* accessor_ptr) const
  {
    const void* placeholder_object_ptr = reinterpret_cast<const void*>(accessor_ptr);
    auto it = _placeholder_buffer_map.find(placeholder_object_ptr);

    if(it == _placeholder_buffer_map.end())
      return nullptr;

    return it->second;
  }

private:
  mutex_class _lock;
  std::unordered_map<const void*, buffer_ptr> _placeholder_buffer_map;
};

class runtime
{
public:

  task_graph& get_task_graph()
  { return _task_graph; }

  const task_graph& get_task_graph() const
  { return _task_graph; }

  placeholder_accessor_tracker&
  get_placeholder_tracker()
  { return _placeholder_tracker; }

  const placeholder_accessor_tracker&
  get_placeholder_tracker() const
  { return _placeholder_tracker; }

private:
  task_graph _task_graph;
  placeholder_accessor_tracker _placeholder_tracker;
};

}
}
}

#endif
