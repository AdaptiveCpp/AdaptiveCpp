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

#ifndef HIPSYCL_ZE_QUEUE_HPP
#define HIPSYCL_ZE_QUEUE_HPP

#include <future>
#include <level_zero/ze_api.h>

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "ze_module.hpp"


namespace hipsycl {
namespace rt {

class ze_hardware_manager;
class ze_queue;
class ze_module_invoker;

class ze_queue : public inorder_queue
{
  friend class ze_module_invoker;
public:
  ze_queue(ze_hardware_manager* hw_manager, std::size_t device_index);

  virtual ~ze_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;

  virtual result submit_memcpy(memcpy_operation&, dag_node_ptr) override;
  virtual result submit_kernel(kernel_operation&, dag_node_ptr) override;
  virtual result submit_prefetch(prefetch_operation &, dag_node_ptr) override;
  virtual result submit_memset(memset_operation&, dag_node_ptr) override;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) override;
  virtual result submit_external_wait_for(dag_node_ptr node) override;

  virtual device_id get_device() const override;
  /// Return native type if supported, nullptr otherwise
  virtual void* get_native_type() const override;

  /// Get a module invoker to launch kernels from module images,
  /// if the backend supports this. Returns nullptr if unsupported.
  virtual module_invoker* get_module_invoker() override;

  ze_command_list_handle_t get_ze_command_list() const {
    return _command_list;
  }

  ze_hardware_manager* get_hardware_manager() const {
    return _hw_manager;
  }

private:
  const std::vector<std::shared_ptr<dag_node_event>>&
  get_enqueued_synchronization_ops() const;
  
  std::vector<ze_event_handle_t>
  get_enqueued_event_handles() const;

  void register_submitted_op(std::shared_ptr<dag_node_event> evt);

  std::shared_ptr<dag_node_event> create_event();

  ze_command_list_handle_t _command_list;
  ze_hardware_manager* _hw_manager;
  std::size_t _device_index;
  ze_module_invoker _module_invoker;

  std::shared_ptr<dag_node_event> _last_submitted_op_event;
  std::vector<std::shared_ptr<dag_node_event>> _enqueued_synchronization_ops;

  std::vector<std::future<void>> _external_waits;
};

}
}

#endif
