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

#include "CL/sycl/backend/backend.hpp"
#include "CL/sycl/detail/buffer.hpp"
#include "CL/sycl/exception.hpp"
#include <cstring>
#include <cassert>

namespace cl {
namespace sycl {
namespace detail {

struct max_aligned_vector
{
  double __attribute__((aligned(128))) x[16];
};

static void* aligned_malloc(size_t size)
{
  size_t num_aligned_units = (size + sizeof(max_aligned_vector) - 1)/sizeof(max_aligned_vector);
  return reinterpret_cast<void*>(new max_aligned_vector [num_aligned_units]);
}

static void* memory_offset(void* ptr, size_t bytes)
{
  return reinterpret_cast<void*>(reinterpret_cast<char*>(ptr)+bytes);
}

buffer_impl::buffer_impl(size_t buffer_size,
                         void* host_ptr)
  : _svm{false},
    _pinned_memory{false},
    _owns_host_memory{false},
    _host_memory{host_ptr},
    _size{buffer_size},
    _write_back{true},
    _write_back_memory{host_ptr}
{
  detail::check_error(hipMalloc(&_buffer_pointer, buffer_size));
}

buffer_impl::buffer_impl(size_t buffer_size,
                         device_alloc_mode device_mode,
                         host_alloc_mode host_alloc_mode)
  : _svm{false},
    _pinned_memory{false},
    _owns_host_memory{false},
    _host_memory{nullptr},
    _size{buffer_size},
    _write_back{true},
    _write_back_memory{nullptr}
{
  if(device_mode == device_alloc_mode::svm &&
     host_alloc_mode != host_alloc_mode::none)
    throw invalid_parameter_error{"buffer_impl: SVM allocation cannot be in conjunction with host allocation"};


  if(device_mode == device_alloc_mode::svm)
#ifdef SYCU_PLATFORM_CUDA
    _svm = true;
#else
    throw unimplemented{"SVM allocation is currently only supported on CUDA"};
#endif

  if(host_alloc_mode != host_alloc_mode::none)
    _owns_host_memory = true;

  if(_svm)
  {
#ifdef SYCU_PLATFORM_CUDA
    if(cudaMallocManaged(&_buffer_pointer, buffer_size) != cudaSuccess)
      throw memory_allocation_error{"Couldn't allocate cuda managed memory"};

    _host_memory = _buffer_pointer;
#endif
  }
  else
  {
    if(_owns_host_memory)
    {
      if(host_alloc_mode == host_alloc_mode::allow_pinned)
      {
        // Try pinned memory
        if(hipHostMalloc(&_host_memory, _size) == hipSuccess)
          _pinned_memory = true;
      }

      if(!_pinned_memory)
        // Pinned memory was either not requested or allocation was
        // unsuccessful
        _host_memory = aligned_malloc(buffer_size);
    }

    _write_back_memory = _host_memory;
    detail::check_error(hipMalloc(&_buffer_pointer, buffer_size));
  }

  _monitor = buffer_state_monitor{this->_svm};
}

buffer_impl::~buffer_impl()
{
  if(_svm)
  {
#ifdef SYCU_PLATFORM_CUDA
    if(_write_back &&
       _write_back_memory != nullptr &&
       _write_back_memory != _buffer_pointer)
      // write back
      memcpy(_write_back_memory, _buffer_pointer, _size);

    cudaFree(this->_buffer_pointer);
#endif
  }
  else
  {
    if(_write_back &&
       _write_back_memory != nullptr)
      hipMemcpy(_write_back_memory,
                _buffer_pointer,
                _size,
                hipMemcpyDeviceToHost);

    hipFree(_buffer_pointer);

    if(_owns_host_memory)
    {
      if(_pinned_memory)
        hipHostFree(_host_memory);
      else
        delete [] reinterpret_cast<max_aligned_vector*>(_host_memory);
    }
  }
}

void* buffer_impl::get_buffer_ptr() const
{
  return this->_buffer_pointer;
}

void* buffer_impl::get_host_ptr() const
{
  return this->_host_memory;
}


bool buffer_impl::is_svm_buffer() const
{
  return _svm;
}

bool buffer_impl::owns_host_memory() const
{
  return _owns_host_memory;
}

bool buffer_impl::owns_pinned_host_memory() const
{
  return _pinned_memory;
}

void buffer_impl::update_host(size_t begin, size_t end) const
{
  if(!_svm)
  {
    if(_host_memory != nullptr)
      detail::check_error(hipMemcpy(memory_offset(_host_memory, begin),
                                    memory_offset(_buffer_pointer, begin),
                                    end-begin,
                                    hipMemcpyDeviceToHost));
  }
}

void buffer_impl::update_host() const
{
  update_host(0, this->_size);
}


void buffer_impl::update_device(size_t begin, size_t end)
{
  if(!_svm)
  {
    if(_host_memory != nullptr)
      detail::check_error(hipMemcpy(memory_offset(_buffer_pointer, begin),
                                    memory_offset(_host_memory, begin),
                                    end-begin,
                                    hipMemcpyHostToDevice));
  }
}

void buffer_impl::update_device()
{
  update_device(0, this->_size);
}

void buffer_impl::write(const void* host_data)
{
  assert(host_data != nullptr);
  if(!_svm)
  {
    detail::check_error(hipMemcpy(_buffer_pointer, host_data,
                                  _size,
                                  hipMemcpyHostToDevice));
  }
  else
  {
    memcpy(_buffer_pointer, _host_data, _size);
  }
}


void buffer_impl::set_write_back(void* ptr)
{
  this->_write_back_memory = ptr;
}

void buffer_impl::enable_write_back(bool writeback)
{
  this->_write_back = writeback;
}

void buffer_impl::execute_buffer_action(buffer_action a)
{
  if(a == buffer_action::update_device)
    this->update_device();
  else if(a == buffer_action::update_host)
    this->update_host();
}

void* buffer_impl::access_host(access::mode m)
{
  execute_buffer_action(_monitor.register_host_access(m));
  return this->get_host_memory();
}

void* buffer_impl::access_device(access::mode m)
{
  execute_buffer_action(_monitor.register_device_access(m));
  return this->get_device_memory();
}

// ----------- buffer_state_monitor ----------------

buffer_state_monitor::buffer_state_monitor(bool is_svm, buffer_state s)
  : _svm{is_svm},
    _state{s}
{}


buffer_action buffer_state_monitor::register_host_access(access::mode m)
{
  if(_svm)
    this->_state = buffer_state::synced;
  else
  {
    const buffer_state old_state = this->_state;
    // if m is written to, we need to set
    // the state to host-ahead
    if(m != access::mode::read)
    {
      // if the buffer was previously in sync,
      // the version on the host will be newer after the access
      if(old_state == buffer_state::synced ||
         old_state == buffer_state::device_ahead)
        // After the registered access, the host will be ahead
        // of the device state
        this->_state = buffer_state::host_ahead;
    }
    else
    {
      // For a read only access, the buffer state on the host
      // is synced if the device was ahead. If the host was ahead,
      // it can remain that way
      if(old_state == buffer_state::device_ahead)
        this->_state = buffer_state::synced;
    }

    // Compare new state against old state to determine required
    // action
    if(old_state != this->_state)
    {
      if(old_state == buffer_state::device_ahead)
        return buffer_action::update_host;
      if(old_state == buffer_state::host_ahead)
        return buffer_action::update_device;
    }
  }
  return buffer_action::none;
}

void buffer_state_monitor::register_device_access(access::mode m)
{
  if(_svm)
    this->_state = buffer_state::synced;
  else
  {
    const buffer_state old_state = this->_state;

    if(m != access::mode::read)
    {
      if(old_state == buffer_state::synced ||
         old_state == buffer_state::host_ahead)
        // After the registered access, the device will be ahead
        // of the host state
        this->_state = buffer_state::device_ahead;
    }
    else
    {
      // For a read only access, the buffer state on the device
      // is synced if the host was ahead. If the device was ahead,
      // nothing needs to change
      if(old_state == buffer_state::host_ahead)
        this->_state = buffer_state::synced;
    }

    if(old_state != this->_state)
    {
      if(old_state == buffer_state::device_ahead)
        return buffer_action::update_host;
      if(old_state == buffer_state::host_ahead)
        return buffer_action::update_device;
    }
  }
  return buffer_action::none;
}


}
}
}
