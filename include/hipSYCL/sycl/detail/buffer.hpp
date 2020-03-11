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

#ifndef HIPSYCL_DETAIL_BUFFER_HPP
#define HIPSYCL_DETAIL_BUFFER_HPP

#include "../backend/backend.hpp"
#include "../types.hpp"
#include "../access.hpp"
#include "../event.hpp"
#include "task_graph.hpp"
#include "stream.hpp"

#include <cstddef>

namespace hipsycl {
namespace sycl {
namespace detail {


enum class host_alloc_mode
{
  svm,
  regular,
  allow_pinned
};

enum class device_alloc_mode
{
  regular,
  svm
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
  buffer_state_monitor(bool is_svm = false);

  buffer_action register_host_access(access::mode m);
  buffer_action register_device_access(access::mode m);

  bool is_host_outdated() const;
  bool is_device_outdated() const;
private:
  bool _svm;

  std::size_t _host_data_version;
  std::size_t _device_data_version;
};

/// Logs operations on the buffer, and calculates
/// dependencies on previous buffer accesses
class buffer_access_log
{
public:
  /// Blocks until all work has completed
  ~buffer_access_log();

  /// Adds a buffer access to the dependency list
  void add_operation(const task_graph_node_ptr& task,
                     access::mode access);

  /// \return whether the buffer is currently in use,
  /// i.e. any operations have been registered.
  bool is_buffer_in_use() const;

  /// Calculates the dependencies required for the specified access mode.
  /// The following rules are used
  vector_class<task_graph_node_ptr>
  calculate_dependencies(access::mode m) const;

  bool is_write_operation_pending() const;

  /// Waits until all dependencies have completed.
  void wait_dependencies();
private:
  struct dependency
  {
    task_graph_node_ptr task;
    access::mode access_mode;
  };

  vector_class<dependency> _operations;
};

class buffer_impl;
using buffer_ptr = shared_ptr_class<buffer_impl>;

class buffer_impl
{
public:

  buffer_impl(size_t buffer_size,
              device_alloc_mode device_mode = device_alloc_mode::regular,
              host_alloc_mode host_alloc_mode = host_alloc_mode::regular);

  buffer_impl(size_t buffer_size,
              void* host_ptr,
              bool is_svm_ptr = false);

  ~buffer_impl();

  void* get_buffer_ptr() const;
  void* get_host_ptr() const;

  void write(const void* host_data, hipStream_t stream, bool async = false);

  bool is_svm_buffer() const;

  bool owns_host_memory() const;
  bool owns_pinned_host_memory() const;

  void set_write_back(void* ptr);

  void enable_write_back(bool writeback);
  
  bool is_writeback_enabled() const;
  void* get_writeback_ptr() const;

  /// Finishes all enqueued host accesses, and executes
  /// possible write-back operations. After a call to this
  /// function, the host-side buffer can be safely released
  /// and it is guaranteed that it won't be accessed anymore,
  /// as long no new host accesses are enqueued.

  void finalize_host(detail::stream_ptr stream);

  static
  task_graph_node_ptr access_host(detail::buffer_ptr buff,
                                  access::mode m,
                                  detail::stream_ptr stream,
                                  async_handler error_handler);

  static
  task_graph_node_ptr access_device(detail::buffer_ptr buff,
                                    access::mode m,
                                    detail::stream_ptr stream,
                                    async_handler error_handler);

  /// Registers an external operation on the buffer in the access log,
  /// such as an explicit copy or a kernel working with the buffer.
  /// The external operation will then be taken account during the dependency
  /// resolution for buffer accesses.
  void register_external_access(const task_graph_node_ptr& task,
                                access::mode m);

private:
  void perform_writeback(detail::stream_ptr stream);

  void update_host(size_t begin, size_t end, hipStream_t stream);
  void update_host(hipStream_t stream);

  void update_device(size_t begin, size_t end, hipStream_t stream);
  void update_device(hipStream_t stream);

  task_state execute_buffer_action(buffer_action a, hipStream_t stream);

  /// Performs an async data transfer if the stream is from
  /// a sycl queue (i.e. not the default stream) and a synchronous
  /// data transfer otherwise.
  void memcpy_d2h(void* host, const void* device, size_t len, hipStream_t stream);

  /// Performs an async data transfer if the stream is from
  /// a sycl queue (i.e. not the default stream) and a synchronous
  /// data transfer otherwise.
  void memcpy_h2d(void* device, const void* host, size_t len, hipStream_t stream);

  bool _svm;
  bool _pinned_memory;
  bool _owns_host_memory;

  void* _buffer_pointer;
  void* _host_memory;

  size_t _size;

  bool _write_back;
  void* _write_back_memory;

  buffer_state_monitor _monitor;
  buffer_access_log _dependency_manager;

  mutex_class _mutex;
};

class buffer_cleanup_trigger
{
public:
  buffer_cleanup_trigger(buffer_ptr buff);
  ~buffer_cleanup_trigger();

  using cleanup_callback = function_class<void ()>;

  // Will be called after the writeback takes place
  void add_cleanup_callback(cleanup_callback callback);
  
  void remove_cleanup_callbacks();
private:
  buffer_ptr _buff;

  vector_class<cleanup_callback> _callbacks;
};



}
}
}

#endif
