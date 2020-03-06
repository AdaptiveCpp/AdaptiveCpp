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

#include "hipSYCL/sycl/backend/backend.hpp"
#include "hipSYCL/sycl/detail/buffer.hpp"
#include "hipSYCL/sycl/exception.hpp"
#include "hipSYCL/sycl/detail/application.hpp"
#include "hipSYCL/sycl/detail/debug.hpp"

#include <cstring>
#include <cassert>
#include <mutex>
#include <algorithm>


namespace hipsycl {
namespace sycl {
namespace detail {
    
    
#ifdef HIPSYCL_SVM_SUPPORTED

static void* svm_alloc(size_t size)
{
#ifdef HIPSYCL_PLATFORM_CUDA
  void* ptr = nullptr;
  if(cudaMallocManaged(&ptr, size) != cudaSuccess)
    throw memory_allocation_error{"Couldn't allocate cuda managed memory"};
  return ptr;
#elif defined(HIPSYCL_PLATFORM_CPU)
  return new char [size];
#else
  raise unimplemented{"Invalid platform for call to svm_alloc()"};
#endif
}

static void svm_free(void* ptr)
{
#ifdef HIPSYCL_PLATFORM_CUDA
  cudaFree(ptr);
#elif defined(HIPSYCL_PLATFORM_CPU)
  delete [] reinterpret_cast<char*>(ptr);
#else
  raise unimplemented{"Invalid platform for call to svm_alloc()"};
#endif
}
  
  
#endif

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
                         void* host_ptr,
                         bool is_svm_ptr)
  : _svm{is_svm_ptr},
    _pinned_memory{false},
    _owns_host_memory{false},
    _host_memory{host_ptr},
    _size{buffer_size},
    _write_back{true},
    _write_back_memory{host_ptr}
{
  if(is_svm_ptr)
  {
#ifdef HIPSYCL_SVM_SUPPORTED
    // If SVM is supported and the provided pointer is an SVM pointer,
    // we can just use it directly as internal data buffer
    _buffer_pointer = host_ptr;
#else
    throw unimplemented{"Attempted to force a buffer to interpret pointer as SVM, but backend does not support SVM pointers"};
#endif
  }
  else
  {
    detail::check_error(hipMalloc(&_buffer_pointer, buffer_size));
  }
  // This tells the buffer state monitor that the host pointer
  // may already have been modified, and guarantees that it will
  // be copied to the device before being used.
  this->_monitor.register_host_access(access::mode::read_write);
}

buffer_impl::buffer_impl(size_t buffer_size,
                         device_alloc_mode device_mode,
                         host_alloc_mode host_mode)
  : _svm{false},
    _pinned_memory{false},
    _owns_host_memory{false},
    _host_memory{nullptr},
    _size{buffer_size},
    _write_back{true},
    _write_back_memory{nullptr}
{
  if((device_mode == device_alloc_mode::svm &&
      host_mode != host_alloc_mode::svm) ||
     (host_mode == host_alloc_mode::svm &&
      device_mode != device_alloc_mode::svm))
    throw invalid_parameter_error{"buffer_impl: SVM allocation must be enabled on both host and device side"};


  if(device_mode == device_alloc_mode::svm)
#ifdef HIPSYCL_SVM_SUPPORTED
    _svm = true;
#else
    throw unimplemented{"SVM allocation is currently only supported on CUDA and CPU backends"};
#endif

  _owns_host_memory = true;

  if(_svm)
  {
#ifdef HIPSYCL_SVM_SUPPORTED
    _buffer_pointer = svm_alloc(buffer_size);
    
    _host_memory = _buffer_pointer;
#endif
  }
  else
  {
    if(_owns_host_memory)
    {
      if(host_mode == host_alloc_mode::allow_pinned)
      {
        // Try pinned memory
        if(hipHostMalloc(&_host_memory, _size, hipHostMallocDefault) == hipSuccess)
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
  _dependency_manager.wait_dependencies();

  if(_svm)
  {
#ifdef HIPSYCL_SVM_SUPPORTED
    // In SVM mode, host memory is the same as buffer memory,
    // so _owns_host_memory tells us if we have allocated buffer
    // memory
    if(_owns_host_memory)
      svm_free(this->_buffer_pointer);
#endif
  }
  else
  {
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

void buffer_impl::perform_writeback(detail::stream_ptr stream)
{
  if(_svm)
  {
#ifdef HIPSYCL_SVM_SUPPORTED
    HIPSYCL_DEBUG_INFO << "buffer_impl: Running implicit SVM write-back"
                       << std::endl;
    if(_write_back &&
       _write_back_memory != nullptr &&
       _write_back_memory != _buffer_pointer)
      // write back
      memcpy(_write_back_memory, _buffer_pointer, _size);
#endif
  }
  else
  {
    if(_write_back &&
       _write_back_memory != nullptr)
    {

      task_graph_node_ptr node;

      {
        
        HIPSYCL_DEBUG_INFO << "buffer_impl: Preparing write-back"
                           << std::endl;
        std::lock_guard<mutex_class> lock(_mutex);

        task_graph& tg = detail::application::get_task_graph();

        auto dependencies =
            _dependency_manager.calculate_dependencies(access::mode::read);

        // It's fine to capture this here, since we will wait
        // for this task anyway.
        auto task = [this, stream] () -> task_state{

          // If host memory is outdated, we need to perform a device->host
          // copy to the writeback memory buffer
          if(_monitor.is_host_outdated())
          {
            HIPSYCL_DEBUG_INFO << "buffer_impl: Executing async "
                                "Device->Host copy for writeback to host buffer"
                                << std::endl;
                                
            detail::check_error(hipMemcpyAsync(_write_back_memory,
                                              _buffer_pointer,
                                              _size,
                                              hipMemcpyDeviceToHost,
                                              stream->get_stream()));
            
            return task_state::enqueued;
          }
          else if (_write_back_memory != _host_memory) 
          {
            // If host is already up-to-date, we only need a regular memcpy from internal
            // to writeback buffer if we use a separate writeback buffer
            HIPSYCL_DEBUG_INFO << "buffer_impl: Copying host buffer content "
                                  "to separate writeback buffer [host memory is already updated]"
                               << std::endl;

            // TODO Use memory_offset()
            std::copy(reinterpret_cast<char *>(_host_memory),
                      reinterpret_cast<char *>(_host_memory) + _size,
                      reinterpret_cast<char *>(_write_back_memory));
            return task_state::complete;
          } 
          else 
          {
            HIPSYCL_DEBUG_INFO << "buffer_impl: Skipping write-back, device memory and "
                                  "write-back memory are already in sync."
                               << std::endl;
            return task_state::complete;
          }

        };

        node = tg.insert(task,
                         dependencies,
                         stream,
                         stream->get_error_handler());
        // Write-back is logically always a read operation since
        // it is executed at buffer destruction when the buffer cannot
        // be changed anymore
        _dependency_manager.add_operation(node, access::mode::read);
      }

      assert(node != nullptr);
      node->wait();
      assert(node->is_done());
      
    }
    else
      HIPSYCL_DEBUG_INFO << "buffer_impl: Skipping write-back, write-back was disabled "
                            "or target memory is NULL"
                         << std::endl;
  }
}

bool buffer_impl::is_writeback_enabled() const
{
  return _write_back;
}

void* buffer_impl::get_writeback_ptr() const
{
  return _write_back_memory;
}

void buffer_impl::finalize_host(detail::stream_ptr stream)
{
  // Additionally, if we use host memory that is not
  // managed by the buffer object, we must wait until
  // all operations on the buffer have finished. If the
  // host buffer is allocated by an external object, it is
  // possible that this object (e.g. std::vector) goes out
  // of scope after the buffer (but not buffer_impl) object
  // is destroyed.
  if(!_owns_host_memory)
    _dependency_manager.wait_dependencies();

  // Writeback must be triggered in any case
  perform_writeback(stream);
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
    assert(_host_memory != nullptr);
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
    assert(_host_memory != nullptr);
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

void buffer_impl::write(const void* host_data, hipStream_t stream, bool async)
{
  std::lock_guard<mutex_class> lock(_mutex);

  assert(host_data != nullptr);
  if(!_svm)
  {
    this->memcpy_h2d(_buffer_pointer, host_data, _size, stream);
    if(!async)
      detail::check_error(hipStreamSynchronize(stream));
  }
  else
  {
    memcpy(_buffer_pointer, host_data, _size);
  }
}


void buffer_impl::set_write_back(void* ptr)
{
  std::lock_guard<mutex_class> lock(_mutex);
  this->_write_back_memory = ptr;
}

void buffer_impl::enable_write_back(bool writeback)
{
  std::lock_guard<mutex_class> lock(_mutex);
  this->_write_back = writeback;
}

task_state
buffer_impl::execute_buffer_action(buffer_action a, hipStream_t stream)
{
  if(a != buffer_action::none)
  {
    if(a == buffer_action::update_device)
      this->update_device(stream);
    else if(a == buffer_action::update_host)
      this->update_host(stream);

    return task_state::enqueued;
  }
  
  HIPSYCL_DEBUG_INFO << "buffer_impl: <Optimizing buffer action to no-op>"
                    << std::endl;
  return task_state::complete;
}


void buffer_impl::memcpy_d2h(void* host, const void* device, size_t len, hipStream_t stream)
{
  HIPSYCL_DEBUG_INFO << "buffer_impl: Executing async "
                        "Device->Host copy of size "
                     << len
                     << std::endl;

  detail::check_error(hipMemcpyAsync(host, device, len,
                                     hipMemcpyDeviceToHost, stream));

}

void buffer_impl::memcpy_h2d(void* device, const void* host, size_t len, hipStream_t stream)
{
  HIPSYCL_DEBUG_INFO << "buffer_impl: Executing async "
                        "Host->Device copy of size " 
                     << len
                     << std::endl;

  detail::check_error(hipMemcpyAsync(device, host, len,
                                     hipMemcpyHostToDevice, stream));
}

task_graph_node_ptr
buffer_impl::access_host(detail::buffer_ptr buff,
                         access::mode m,
                         detail::stream_ptr stream,
                         async_handler error_handler)
{
  std::lock_guard<mutex_class> lock(buff->_mutex);

  task_graph& tg = detail::application::get_task_graph();

  auto dependencies = buff->_dependency_manager.calculate_dependencies(m);

  auto task = [buff, m, stream] () -> task_state {
    return buff->execute_buffer_action(
          buff->_monitor.register_host_access(m),
          stream->get_stream());
  };

  task_graph_node_ptr node = tg.insert(task, dependencies, stream, error_handler);
  buff->_dependency_manager.add_operation(node, m);

  return node;
}

task_graph_node_ptr
buffer_impl::access_device(detail::buffer_ptr buff,
                           access::mode m,
                           detail::stream_ptr stream,
                           async_handler error_handler)
{
  std::lock_guard<mutex_class> lock(buff->_mutex);

  task_graph& tg = detail::application::get_task_graph();

  auto dependencies = buff->_dependency_manager.calculate_dependencies(m);

  auto task = [buff, m, stream] () -> task_state {
    return buff->execute_buffer_action(
          buff->_monitor.register_device_access(m),
          stream->get_stream());

  };

  task_graph_node_ptr node = tg.insert(task, dependencies, stream, error_handler);
  buff->_dependency_manager.add_operation(node, m);

  return node;
}

void
buffer_impl::register_external_access(const task_graph_node_ptr& task,
                                      access::mode m)
{
  std::lock_guard<mutex_class> lock(_mutex);
  this->_dependency_manager.add_operation(task, m);
}

void* buffer_impl::get_buffer_ptr() const
{
  return _buffer_pointer;
}

void* buffer_impl::get_host_ptr() const
{
  return _host_memory;
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
    HIPSYCL_DEBUG_INFO << "buffer_state_info: host access, current data versions: host-side is @"
                       << _host_data_version
                       << ", device side is @"
                       << _device_data_version
                       << std::endl;

    // Make sure host is up-to-date before reading
    bool copy_required = this->is_host_outdated();
    bool read_only = (m == access::mode::read);

    // If we are discarding the previous content anyway,
    // a data transfer is never required
    if(m == access::mode::discard_write ||
       m == access::mode::discard_read_write)
      copy_required = false;

    size_t version_bump = read_only ? 0 : 1;
    _host_data_version =
      std::max(_host_data_version, _device_data_version) + version_bump;

    HIPSYCL_DEBUG_INFO << "buffer_state_info: (new state: host @"
                       << _host_data_version << ", device @"
                       << _device_data_version << ")"
                       << std::endl;

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
     HIPSYCL_DEBUG_INFO << "buffer_state_info: device access, current data versions: host-side is @"
                       << _host_data_version
                       << ", device side is @"
                       << _device_data_version
                       << std::endl;

    // Make sure device is up-to-date before reading
    bool copy_required = this->is_device_outdated();
    bool read_only = (m == access::mode::read);

    // If we are discarding the previous content anyway,
    // a data transfer is never required
    if(m == access::mode::discard_write ||
       m == access::mode::discard_read_write)
      copy_required = false;

    size_t version_bump = read_only ? 0 : 1;
    _device_data_version =
      std::max(_host_data_version, _device_data_version) + version_bump;

    HIPSYCL_DEBUG_INFO << "buffer_state_info: (new state: host @"
                       << _host_data_version << ", device @"
                       << _device_data_version << ")"
                       << std::endl;

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

// -------------- buffer_access_log ----------------


void buffer_access_log::add_operation(const task_graph_node_ptr& task,
                                      access::mode access)
{  
  _operations.push_back({task, access});

  for(auto it = _operations.begin();
      it != _operations.end();)
  {
    if(it->task->is_done())
      it = _operations.erase(it);
    else
      ++it;
  }
}

bool buffer_access_log::is_buffer_in_use() const
{
  return _operations.size() > 0;
}

buffer_access_log::~buffer_access_log()
{
  for(auto& op : _operations)
    op.task->wait();
}


vector_class<task_graph_node_ptr>
buffer_access_log::calculate_dependencies(access::mode m) const
{
  vector_class<task_graph_node_ptr> deps;

  if(m != access::mode::read)
  {
    // Write operations need to wait until all previous
    // reads and writes have finished to guarantee consistency
    for(auto op : _operations)
      deps.push_back(op.task);
  }
  else
  {
    // Read-only operations do not need to depend on previous
    // read operations
    for(auto op: _operations)
      if(op.access_mode != access::mode::read)
        deps.push_back(op.task);
  }

  return deps;
}

bool
buffer_access_log::is_write_operation_pending() const
{
  for(auto op : _operations)
    if(op.access_mode != access::mode::read &&
       !op.task->is_done())
      return true;
  return false;
}

void
buffer_access_log::wait_dependencies()
{
  for(auto op : _operations)
    op.task->wait();
}

///////////////////// buffer_writeback_trigger ///////////////////

buffer_cleanup_trigger::buffer_cleanup_trigger(buffer_ptr buff)
  : _buff{buff}
{}

buffer_cleanup_trigger::~buffer_cleanup_trigger()
{
  HIPSYCL_DEBUG_INFO << "buffer_cleanup_trigger: Buffer went out of scope,"
                        " triggering cleanup" << std::endl;
  detail::stream_ptr stream = std::make_shared<detail::stream_manager>();
  _buff->finalize_host(stream);

  for(auto callback : _callbacks)
    callback();
}

void
buffer_cleanup_trigger::add_cleanup_callback(
    buffer_cleanup_trigger::cleanup_callback callback)
{
  this->_callbacks.push_back(callback);
}

void 
buffer_cleanup_trigger::remove_cleanup_callbacks()
{
  this->_callbacks.clear();
}

}
}
}
