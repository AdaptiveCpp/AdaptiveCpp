/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_OCL_QUEUE_HPP
#define HIPSYCL_OCL_QUEUE_HPP

#include <CL/opencl.hpp>
#include <mutex>

#include "../executor.hpp"
#include "../inorder_queue.hpp"

#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"

namespace hipsycl {
namespace rt {

class ocl_hardware_manager;

class ocl_queue : public inorder_queue
{
public:
  ocl_queue(ocl_hardware_manager* hw_manager, std::size_t device_index);
  
  ocl_queue(const ocl_queue&) = delete;
  ocl_queue& operator=(const ocl_queue&) = delete;

  virtual ~ocl_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  virtual result submit_memcpy(memcpy_operation&, dag_node_ptr) override;
  virtual result submit_kernel(kernel_operation&, dag_node_ptr) override;
  virtual result submit_prefetch(prefetch_operation &, dag_node_ptr) override;
  virtual result submit_memset(memset_operation&, dag_node_ptr) override;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(dag_node_ptr evt) override;
  virtual result submit_external_wait_for(dag_node_ptr node) override;

  virtual result wait() override;

  virtual device_id get_device() const override;
  /// Return native type if supported, nullptr otherwise
  virtual void* get_native_type() const override;

  virtual result query_status(inorder_queue_status& status) override;

  ocl_hardware_manager* get_hardware_manager() const;

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &kernel_name, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const kernel_configuration &config);

private:
  void register_submitted_op(cl::Event);

  // These member variables have to be thread-safe.
  ocl_hardware_manager* _hw_manager;
  const std::size_t _device_index;

  cl::CommandQueue _queue;
  ocl_sscp_code_object_invoker _sscp_invoker;
  worker_thread _host_worker;

  std::shared_ptr<kernel_cache> _kernel_cache;

  // Non-thread safe state should go here
  struct protected_state {
  public:
    auto get_most_recent_event() const {
      std::lock_guard<std::mutex> lock {_mutex};
      return _most_recent_event;
    }

    template<class T>
    void set_most_recent_event(const T& x) {
      std::lock_guard<std::mutex> lock {_mutex};
      _most_recent_event = x;
    }
  private:
    std::shared_ptr<dag_node_event> _most_recent_event = nullptr;
    mutable std::mutex _mutex;
  };

  protected_state _state;
};

}
}

#endif
