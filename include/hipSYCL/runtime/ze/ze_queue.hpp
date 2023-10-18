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
#include <mutex>
#include <level_zero/ze_api.h>

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "ze_code_object.hpp"


namespace hipsycl {
namespace rt {

class ze_hardware_manager;
class ze_queue;

class ze_queue : public inorder_queue
{
public:
  ze_queue(ze_hardware_manager* hw_manager, std::size_t device_index);

  virtual ~ze_queue();

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

  ze_command_list_handle_t get_ze_command_list() const {
    return _command_list;
  }

  ze_hardware_manager* get_hardware_manager() const {
    return _hw_manager;
  }

  result submit_multipass_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &backend_kernel_name, const rt::range<3> &grid_size,
      const rt::range<3> &block_size, unsigned dynamic_shared_mem,
      void **kernel_args, const std::size_t *arg_sizes, std::size_t num_args);

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &kernel_name, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const glue::kernel_configuration &config);

private:
  const std::vector<std::shared_ptr<dag_node_event>>&
  get_enqueued_synchronization_ops() const;
  
  std::vector<ze_event_handle_t>
  get_enqueued_event_handles() const;

  void register_submitted_op(std::shared_ptr<dag_node_event> evt);

  std::shared_ptr<dag_node_event> create_event();

  ze_command_list_handle_t _command_list;
  ze_hardware_manager* _hw_manager;
  const std::size_t _device_index;
  ze_multipass_code_object_invoker _multipass_code_object_invoker;
  ze_sscp_code_object_invoker _sscp_code_object_invoker;

  std::shared_ptr<dag_node_event> _last_submitted_op_event;
  std::vector<std::shared_ptr<dag_node_event>> _enqueued_synchronization_ops;

  std::vector<std::future<void>> _external_waits;

  // Most L0 API functions that add to a command list are not thread-safe.
  // Since most of the public API functions of this class do exactly that,
  // arguably the best strategy to achieve thread-safety is to just have a mutex
  // and lock in every public function.
  std::mutex _mutex;
};

}
}

#endif
