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
    {
      if((_write_back_memory == _host_memory && _monitor.is_host_outdated())
       || _write_back_memory != _host_memory)
        hipMemcpy(_write_back_memory,
                  _buffer_pointer,
                  _size,
                  hipMemcpyDeviceToHost);
    }

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

void buffer_impl::update_host(size_t begin, size_t end, hipStream_t stream)
{
  if(!_svm)
  {
    if(_host_memory != nullptr)
      this->memcpy_d2h(memory_offset(_host_memory, begin),
                       memory_offset(_buffer_pointer, begin),
                       end-begin,
                       stream);
  }
}

void buffer_impl::update_host(hipStream_t stream)
{
  update_host(0, this->_size, stream);
}


void buffer_impl::update_device(size_t begin, size_t end, hipStream_t stream)
{
  if(!_svm)
  {
    if(_host_memory != nullptr)
      this->memcpy_h2d(memory_offset(_buffer_pointer, begin),
                       memory_offset(_host_memory, begin),
                       end-begin,
                       stream);
  }
}

void buffer_impl::update_device(hipStream_t stream)
{
  update_device(0, this->_size, stream);
}

void buffer_impl::write(const void* host_data, hipStream_t stream)
{
  assert(host_data != nullptr);
  if(!_svm)
    this->memcpy_h2d(_buffer_pointer, host_data, _size, stream);
  else
  {
    memcpy(_buffer_pointer, _host_memory, _size);
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

void buffer_impl::execute_buffer_action(buffer_action a, hipStream_t stream)
{
  if(a == buffer_action::update_device)
    this->update_device(stream);
  else if(a == buffer_action::update_host)
    this->update_host(stream);
}

void* buffer_impl::access_host(access::mode m, hipStream_t stream)
{
  execute_buffer_action(_monitor.register_host_access(m), stream);
  return _host_memory;
}

void* buffer_impl::access_device(access::mode m, hipStream_t stream)
{
  execute_buffer_action(_monitor.register_device_access(m), stream);
  return _buffer_pointer;
}

void buffer_impl::memcpy_d2h(void* host, const void* device, size_t len, hipStream_t stream)
{
  detail::check_error(hipMemcpyAsync(host, device, len,
                                     hipMemcpyDeviceToHost, stream));
  detail::check_error(hipStreamSynchronize(stream));
}

void buffer_impl::memcpy_h2d(void* device, const void* host, size_t len, hipStream_t stream)
{
  detail::check_error(hipMemcpyAsync(device, host, len,
                                     hipMemcpyHostToDevice, stream));
  detail::check_error(hipStreamSynchronize(stream));
}

// ----------- buffer_state_monitor ----------------

buffer_state_monitor::buffer_state_monitor(bool is_svm)
  : _svm{is_svm}, _host_data_version{0}, _device_data_version{0}
{}


buffer_action buffer_state_monitor::register_host_access(access::mode m)
{
  if(_svm)
  {
    // With svm, host and device are always in sync
    _host_data_version = 0;
    _device_data_version = 0;
  }
  else
  {

    // Make sure host is up-to-date before reading
    bool copy_required = _host_data_version < _device_data_version;

    if(m != access::mode::read)
      _host_data_version++;
    else
      _host_data_version = _device_data_version;

    if(copy_required)
      return buffer_action::update_host;

  }
  return buffer_action::none;
}

buffer_action buffer_state_monitor::register_device_access(access::mode m)
{
  if(_svm)
  {
    // With svm, host and device are always in sync
    _host_data_version = 0;
    _device_data_version = 0;
  }
  else
  {

    // Make sure device is up-to-date before reading
    bool copy_required = _device_data_version < _host_data_version;

    if(m != access::mode::read)
      _device_data_version++;
    else
      _device_data_version = _host_data_version;

    if(copy_required)
      return buffer_action::update_device;

  }
  return buffer_action::none;
}

bool buffer_state_monitor::is_host_outdated() const
{
  return this->_host_data_version < this->_device_data_version;
}

bool buffer_state_monitor::is_device_outdated() const
{
  return this->_device_data_version < this->_host_data_version;
}

}
}
}
