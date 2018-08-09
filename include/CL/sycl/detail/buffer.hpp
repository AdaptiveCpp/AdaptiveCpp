/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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

#ifndef SYCU_DETAIL_BUFFER_HPP
#define SYCU_DETAIL_BUFFER_HPP

#include "../backend/backend.hpp"
#include "../types.hpp"
#include "../access.hpp"

namespace cl {
namespace sycl {
namespace detail {


enum class host_alloc_mode
{
  none,
  regular,
  allow_pinned
};

enum class device_alloc_mode
{
  regular,
  svm
};

enum class buffer_state
{
  synced,
  device_ahead,
  host_ahead
};

enum class buffer_action
{
  none,
  update_device,
  update_host
};

class buffer_state_monitor
{
public:
  buffer_state_monitor(bool is_svm = false, buffer_state s = buffer_state::synced);

  buffer_action register_host_access(access::mode m);
  buffer_action register_device_access(access::mode m);

private:
  bool _svm;
  buffer_state _state;
};

class buffer_impl
{
public:

  buffer_impl(size_t buffer_size,
              device_alloc_mode device_mode = device_alloc_mode::regular,
              host_alloc_mode host_alloc_mode = host_alloc_mode::regular);

  buffer_impl(size_t buffer_size,
              void* host_ptr);

  ~buffer_impl();

  void* get_buffer_ptr() const;
  void* get_host_ptr() const;

  void write(const void* host_data);

  void update_host(size_t begin, size_t end) const;
  void update_host() const;

  void update_device(size_t begin, size_t end);
  void update_device();

  bool is_svm_buffer() const;

  bool owns_host_memory() const;
  bool owns_pinned_host_memory() const;

  void set_write_back(void* ptr);
  void enable_write_back(bool writeback);

  void* access_host(access::mode m);
  void* access_device(access::mode m);
private:
  void execute_buffer_action(buffer_action a);

  bool _svm;
  bool _pinned_memory;
  bool _owns_host_memory;

  void* _buffer_pointer;
  void* _host_memory;

  size_t _size;

  bool _write_back;
  void* _write_back_memory;

  buffer_state_monitor _monitor;
};


using buffer_ptr = shared_ptr_class<detail::buffer_impl>;

}
}
}

#endif
