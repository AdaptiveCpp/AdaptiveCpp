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

#include <cassert>
#include <chrono>
#include <future>
#include <utility>
#include <level_zero/ze_api.h>

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/ze/ze_module.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_event.hpp"
#include "hipSYCL/runtime/util.hpp"

namespace hipsycl {
namespace rt {

ze_queue::ze_queue(ze_hardware_manager *hw_manager, std::size_t device_index)
    : _hw_manager{hw_manager}, _device_index{device_index}, _module_invoker{
                                                                this} {
  assert(hw_manager);

  ze_hardware_context *hw_context =
      cast<ze_hardware_context>(hw_manager->get_device(device_index));
  
  assert(hw_context);

  ze_command_queue_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
  desc.pNext = nullptr;
  desc.ordinal = 0; // TODO: Query command queue groups and select
                    // appropriate group
  desc.index = 0;
  desc.flags = ZE_COMMAND_QUEUE_FLAG_EXPLICIT_ONLY;
  desc.mode  = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS; 
  desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

  ze_result_t err = zeCommandListCreateImmediate(hw_context->get_ze_context(),
                                                 hw_context->get_ze_device(),
                                                 &desc, &_command_list);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not create immediate command list",
                   error_code{"ze", static_cast<int>(err)}});
  }
}

ze_queue::~ze_queue() {

  ze_result_t err = zeCommandListDestroy(_command_list);
  if(err != ZE_RESULT_SUCCESS) {
    register_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not destroy immediate command list",
                   error_code{"ze", static_cast<int>(err)}});
  }
}

std::shared_ptr<dag_node_event> ze_queue::create_event() {

  ze_event_handle_t evt;
  ze_event_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_EVENT_DESC;
  desc.pNext = nullptr;
  // desc.index (index within the pool) is set by allocate_event()
  desc.signal = ZE_EVENT_SCOPE_FLAG_HOST;
  desc.wait = ZE_EVENT_SCOPE_FLAG_HOST;

  ze_event_pool_manager *pool_mgr =
      _hw_manager->get_event_pool_manager(_device_index);
  assert(pool_mgr);

  std::shared_ptr<ze_event_pool_handle_t> pool =
      pool_mgr->allocate_event(desc.index);

  ze_result_t err = zeEventCreate(*pool, &desc, &evt);

  if(err != ZE_RESULT_SUCCESS) {
    register_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not create event",
                   error_code{"ze", static_cast<int>(err)}});

    return nullptr;
  }

  return std::make_shared<ze_node_event>(evt, pool);
}

std::shared_ptr<dag_node_event> ze_queue::insert_event() {
  if(!_last_submitted_op_event) {
    auto evt = create_event();
    ze_result_t err = zeEventHostSignal(
        static_cast<ze_node_event *>(evt.get())->get_event_handle());
    return evt;
  }
  return _last_submitted_op_event;
}

result ze_queue::submit_memcpy(memcpy_operation& op, dag_node_ptr node) {

  // TODO We could probably unify some of the logic here between
  // ze/cuda/hip backends
  device_id source_dev = op.source().get_device();
  device_id dest_dev = op.dest().get_device();

  assert(op.source().get_access_ptr());
  assert(op.dest().get_access_ptr());

  range<3> transfer_range = op.get_num_transferred_elements();

  int dimension = 0;
  if (transfer_range[0] > 1)
    dimension = 3;
  else if (transfer_range[1] > 1)
    dimension = 2;
  else
    dimension = 1;

  // If we transfer the entire buffer, treat it as 1D memcpy for performance.
  // TODO: The same optimization could also be applied for the general case
  // when regions are contiguous
  if (op.get_num_transferred_elements() == op.source().get_allocation_shape() &&
      op.get_num_transferred_elements() == op.dest().get_allocation_shape() &&
      op.source().get_access_offset() == id<3>{} &&
      op.dest().get_access_offset() == id<3>{})
    dimension = 1;

  assert(dimension >= 1 && dimension <= 3);

  std::shared_ptr<dag_node_event> completion_evt = create_event();
  std::vector<ze_event_handle_t> wait_events = get_enqueued_event_handles();

  if(dimension == 1) {
    ze_result_t err = zeCommandListAppendMemoryCopy(
        _command_list, op.dest().get_access_ptr(), op.source().get_access_ptr(),
        op.get_num_transferred_bytes(),
        static_cast<ze_node_event *>(completion_evt.get())->get_event_handle(),
        static_cast<uint32_t>(wait_events.size()), wait_events.data());

    if(err != ZE_RESULT_SUCCESS) {
      return make_error(
          __hipsycl_here(),
          error_info{"ze_queue: zeCommandListAppendMemoryCopy() failed",
                     error_code{"ze", static_cast<int>(err)}});
    }
  } else {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ze_queue: Multidimensional memory copies are not yet supported.",
            error_type::unimplemented});
  }

  register_submitted_op(completion_evt);
  return make_success();
}

result ze_queue::submit_kernel(kernel_operation& op, dag_node_ptr node) {
  rt::backend_kernel_launcher *l = 
      op.get_launcher().find_launcher(backend_id::level_zero);
  
  if (!l)
    return make_error(__hipsycl_here(),
                      error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);
  l->invoke(node.get());

  return make_success();
}

result ze_queue::submit_prefetch(prefetch_operation &, dag_node_ptr node) {
  return make_success();
}

result ze_queue::submit_memset(memset_operation&, dag_node_ptr node) {
  return make_success();
}
  
result ze_queue::submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) {
  _enqueued_synchronization_ops.push_back(evt);
  return make_success();
}

result ze_queue::submit_external_wait_for(dag_node_ptr node) {
  // Clean up old futures before adding new ones
  _external_waits.erase(
      std::remove_if(_external_waits.begin(), _external_waits.end(),
                     [](const std::future<void> &f) {
                       return f.wait_for(std::chrono::seconds(0)) ==
                              std::future_status::ready;
                     }),
      _external_waits.end());

  auto evt = create_event();
  _enqueued_synchronization_ops.push_back(evt);

  std::future<void> f = std::async(std::launch::async, [evt, node](){
    node->wait();
    ze_result_t err = zeEventHostSignal(
        static_cast<ze_node_event *>(evt.get())->get_event_handle());
    if(err != ZE_RESULT_SUCCESS) {
      register_error(
          __hipsycl_here(),
          error_info{"ze_queue: Couldn't signal completion of external event",
                     error_code{"ze", static_cast<int>(err)}});
    }
  });

  _external_waits.push_back(std::move(f));

  return make_success();
}

device_id ze_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}

void* ze_queue::get_native_type() const {
  return static_cast<void*>(_command_list);
}

module_invoker* ze_queue::get_module_invoker() {
  return &_module_invoker;
}

const std::vector<std::shared_ptr<dag_node_event>>&
ze_queue::get_enqueued_synchronization_ops() const {
  return _enqueued_synchronization_ops;
}

std::vector<ze_event_handle_t>
ze_queue::get_enqueued_event_handles() const {
  const auto& wait_events = get_enqueued_synchronization_ops();

  std::vector<ze_event_handle_t> evts;
  if(!wait_events.empty()) {
    evts.reserve(wait_events.size());
    for(std::size_t i = 0; i < wait_events.size(); ++i) {
      evts[i] = static_cast<ze_node_event *>(wait_events[i].get())
                    ->get_event_handle();
    }
  }
  return evts;
}

void ze_queue::register_submitted_op(std::shared_ptr<dag_node_event> evt) {
  _last_submitted_op_event = evt;
  _enqueued_synchronization_ops.clear();
  _enqueued_synchronization_ops.push_back(evt);
}


}
}

