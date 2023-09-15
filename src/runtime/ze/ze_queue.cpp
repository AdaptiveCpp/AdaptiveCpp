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
#include <vector>

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/ze/ze_code_object.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_event.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirvFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"

#endif

namespace hipsycl {
namespace rt {

namespace {

result submit_ze_kernel(ze_kernel_handle_t kernel,
                        ze_command_list_handle_t command_list,
                        ze_event_handle_t completion_evt,
                        const std::vector<ze_event_handle_t>& wait_events, 
                        const rt::range<3> &group_size,
                        const rt::range<3> &num_groups, void **kernel_args,
                        const std::size_t *arg_sizes, std::size_t num_args,
                        // If non-null, will be used to check whether kernel args
                        // are pointers, and if so, check for null pointers
                        const hcf_kernel_info *info = nullptr) {

  HIPSYCL_DEBUG_INFO << "ze_queue: Configuring kernel launch for group size "
                     << group_size[0] << " " << group_size[1] << " "
                     << group_size[2] << std::endl;
  ze_result_t err =
      zeKernelSetGroupSize(kernel, static_cast<uint32_t>(group_size[0]),
                           static_cast<uint32_t>(group_size[1]),
                           static_cast<uint32_t>(group_size[2]));
  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_module_invoker: Could not set kernel group size",
                   error_code{"ze", static_cast<int>(err)}});
  }

  HIPSYCL_DEBUG_INFO << "ze_queue: Configuring kernel launch for group count "
                     << num_groups[0] << " " << num_groups[1] << " "
                     << num_groups[2] << std::endl;
  ze_group_count_t group_count;
  group_count.groupCountX = static_cast<uint32_t>(num_groups[0]);
  group_count.groupCountY = static_cast<uint32_t>(num_groups[1]);
  group_count.groupCountZ = static_cast<uint32_t>(num_groups[2]);

  for(std::size_t i = 0; i < num_args; ++i ){
    HIPSYCL_DEBUG_INFO << "ze_module_invoker: Setting kernel argument " << i
                       << " of size " << arg_sizes[i] << " at " << kernel_args[i]
                       << std::endl;

    auto points_to_nullptr = [](void* arg) -> bool {
      void* ptr = nullptr;
      std::memcpy(&ptr, arg, sizeof(void*));
      return ptr == nullptr;
    };

    if (info && (info->get_argument_type(i) == hcf_kernel_info::pointer) &&
        points_to_nullptr(kernel_args[i])) {
      // Level Zero absolutely does not like when nullptrs are passed
      // in as values at kernel_args[i] - it validates that those are non-null.
      // So instead, we need to set the argument to zeKernelSetArgumentValue
      // to null.
      err = zeKernelSetArgumentValue(
          kernel, i, static_cast<uint32_t>(arg_sizes[i]), nullptr);
    } else {
      err = zeKernelSetArgumentValue(
          kernel, i, static_cast<uint32_t>(arg_sizes[i]), kernel_args[i]);
    }
    if(err != ZE_RESULT_SUCCESS) {
      return make_error(
          __hipsycl_here(),
          error_info{"ze_module_invoker: Could not set kernel argument",
                     error_code{"ze", static_cast<int>(err)}});
    }
  }

  // This is necessary for USM pointers, which hipSYCL *always*
  // relies on.
  err = zeKernelSetIndirectAccess(kernel,
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED);

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ze_module_invoker: Could not set indirect access flags",
                     error_code{"ze", static_cast<int>(err)}});
  }

  HIPSYCL_DEBUG_INFO << "ze_module_invoker: Submitting kernel!" << std::endl;
  err = zeCommandListAppendLaunchKernel(
      command_list, kernel, &group_count, completion_evt,
      static_cast<uint32_t>(wait_events.size()),
      const_cast<ze_event_handle_t *>(wait_events.data()));

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_module_invoker: Kernel launch failed",
                   error_code{"ze", static_cast<int>(err)}});
  }

  return make_success();
}

}

ze_queue::ze_queue(ze_hardware_manager *hw_manager, std::size_t device_index)
    : _hw_manager{hw_manager}, _device_index{device_index},
      _multipass_code_object_invoker{this}, _sscp_code_object_invoker{this} {
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

std::shared_ptr<dag_node_event> ze_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<ze_event_handle_t, ze_node_event>>(
      this);
}


std::shared_ptr<dag_node_event> ze_queue::insert_event() {
  std::lock_guard<std::mutex> lock{_mutex};

  if(!_last_submitted_op_event) {
    auto evt = create_event();
    ze_result_t err = zeEventHostSignal(
        static_cast<ze_node_event *>(evt.get())->get_event_handle());
    return evt;
  }
  return _last_submitted_op_event;
}

result ze_queue::submit_memcpy(memcpy_operation& op, dag_node_ptr node) {
  std::lock_guard<std::mutex> lock{_mutex};

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
  std::lock_guard<std::mutex> lock{_mutex};

  rt::backend_kernel_launcher *l = 
      op.get_launcher().find_launcher(backend_id::level_zero);
  
  if (!l)
    return make_error(__hipsycl_here(),
                      error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);
  
  rt::backend_kernel_launch_capabilities cap;
  
  cap.provide_multipass_invoker(&_multipass_code_object_invoker);
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);

  l->set_backend_capabilities(cap);

  l->invoke(node.get(), op.get_launcher().get_kernel_configuration());

  return make_success();
}

result ze_queue::submit_prefetch(prefetch_operation &, dag_node_ptr node) {
  return make_success();
}

result ze_queue::submit_memset(memset_operation& op, dag_node_ptr node) {
  std::lock_guard<std::mutex> lock{_mutex};

  std::shared_ptr<dag_node_event> completion_evt = create_event();
  std::vector<ze_event_handle_t> wait_events = get_enqueued_event_handles();
  
  auto pattern = op.get_pattern();
  ze_result_t err = zeCommandListAppendMemoryFill(
      _command_list, op.get_pointer(), &pattern, sizeof(decltype(pattern)),
      op.get_num_bytes(),
      static_cast<ze_node_event *>(completion_evt.get())->get_event_handle(),
      static_cast<uint32_t>(wait_events.size()), wait_events.data());

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ze_queue: zeCommandListAppendMemoryFill() failed",
                     error_code{"ze", static_cast<int>(err)}});
  }

  register_submitted_op(completion_evt);

  return make_success();
}

result ze_queue::wait() {
  std::shared_ptr<dag_node_event> _last_event;
  {
    std::lock_guard<std::mutex> lock{_mutex};
    _last_event = _last_submitted_op_event;
  }
  if(_last_event)
    _last_event->wait();
  return make_success();
}

result ze_queue::submit_queue_wait_for(dag_node_ptr node) {
  std::lock_guard<std::mutex> lock{_mutex};

  auto evt = node->get_event();
  _enqueued_synchronization_ops.push_back(evt);
  return make_success();
}

result ze_queue::submit_external_wait_for(dag_node_ptr node) {
  std::lock_guard<std::mutex> lock{_mutex};

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

result ze_queue::query_status(inorder_queue_status &status) {
  std::lock_guard<std::mutex> lock{_mutex};
  status = inorder_queue_status{this->_last_submitted_op_event->is_complete()};
  return make_success();
}

device_id ze_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}

void* ze_queue::get_native_type() const {
  return static_cast<void*>(_command_list);
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

result ze_queue::submit_multipass_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    const std::string &backend_kernel_name, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned dynamic_shared_mem,
    void **kernel_args, const std::size_t* arg_sizes, std::size_t num_args) {

  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t* kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  ze_hardware_context *hw_ctx = static_cast<ze_hardware_context *>(
      _hw_manager->get_device(_device_index));
  ze_context_handle_t ctx = hw_ctx->get_ze_context();
  ze_device_handle_t dev = hw_ctx->get_ze_device();

  auto code_object_selector = [&](const code_object *candidate) -> bool {
    if ((candidate->managing_backend() != backend_id::level_zero) ||
        (candidate->source_compilation_flow() !=
         compilation_flow::explicit_multipass) ||
        (candidate->state() != code_object_state::executable))
      return false;

    const ze_executable_object *obj =
        static_cast<const ze_executable_object *>(candidate);
    return obj->get_ze_device() == dev && obj->get_ze_context() == ctx;
  };

  auto code_object_constructor = [&]() -> code_object* {
    const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
    if(!hcf) {
      HIPSYCL_DEBUG_ERROR << "ze_queue: Could not retrieve SPIR-V HCF object"
                          << std::endl;
      return nullptr;
    }
    assert(hcf->root_node());
    if(!hcf->root_node()->has_binary_data_attached())
      return nullptr;
    
    std::string code;
    if(!hcf->get_binary_attachment(hcf->root_node(), code)) {
      HIPSYCL_DEBUG_ERROR
          << "ze_queue: Could not extract code from HCF node; invalid HCF data?"
          << std::endl;
    }

    ze_executable_object *exec_obj = new ze_executable_object{
        ctx, dev, hcf_object, ze_source_format::spirv, code};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, backend_kernel_name, backend_id::level_zero, hcf_object,
      code_object_selector, code_object_constructor);

  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"ze_queue: Code object construction failed"});
  }

  ze_kernel_handle_t kernel;
  result res = static_cast<const ze_executable_object *>(obj)->get_kernel(
      backend_kernel_name, kernel);
  
  if(!res.is_success())
    return res;

  std::vector<ze_event_handle_t> wait_events =
      get_enqueued_event_handles();
  std::shared_ptr<dag_node_event> completion_evt = create_event();

  auto submission_err = submit_ze_kernel(
      kernel, get_ze_command_list(),
      static_cast<ze_node_event *>(completion_evt.get())->get_event_handle(),
      wait_events, group_size, num_groups, kernel_args, arg_sizes, num_args);

  if(!submission_err.is_success())
    return submission_err;

  register_submitted_op(completion_evt);

  return make_success();
}

result ze_queue::submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &kernel_name, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const glue::kernel_configuration &initial_config) {

#ifdef HIPSYCL_WITH_SSCP_COMPILER

  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t* kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  ze_hardware_context *hw_ctx = static_cast<ze_hardware_context *>(
      _hw_manager->get_device(_device_index));
  ze_context_handle_t ctx = hw_ctx->get_ze_context();
  ze_device_handle_t dev = hw_ctx->get_ze_device();

  // Need to create custom config to ensure we can distinguish other
  // kernels compiled with different values e.g. of local mem allocation size
  glue::kernel_configuration config = initial_config;
  config.set("spirv-dynamic-local-mem-allocation-size", local_mem_size);
  auto configuration_id = config.generate_id();

  const hcf_kernel_info *kernel_info =
      rt::hcf_cache::get().get_kernel_info(hcf_object, kernel_name);
  if(!kernel_info) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_queue: Could not obtain hcf kernel info for kernel " +
            global_kernel_name});
  }

  auto code_object_selector = [&](const code_object *candidate) -> bool {
    if ((candidate->managing_backend() != backend_id::level_zero) ||
        (candidate->source_compilation_flow() != compilation_flow::sscp) ||
        (candidate->state() != code_object_state::executable))
      return false;

    const ze_sscp_executable_object *obj =
        static_cast<const ze_sscp_executable_object *>(candidate);
    
    if(obj->configuration_id() != configuration_id)
      return false;

    return obj->get_ze_device() == dev && obj->get_ze_context() == ctx;
  };

  auto code_object_constructor = [&]() -> code_object* {
    const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        glue::jit::select_image(kernel_info, &kernel_names);

    // Construct SPIR-V translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      std::move(compiler::createLLVMToSpirvTranslator(kernel_names));

    translator->setBuildOption("spirv-dynamic-local-mem-allocation-size",
                               local_mem_size);
    // Note: As soonst as this is not enabled unconditionally,
    // make sure to also set it in the kernel config!
    translator->setBuildFlag("enable-intel-llvm-spirv-options");
    
    // Lower kernels to SPIR-V
    std::string compiled_image;
    auto err = glue::jit::compile(translator.get(),
        hcf, selected_image_name, config, compiled_image);
    
    if(!err.is_success()) {
      register_error(err);
      return nullptr;
    }

    ze_sscp_executable_object *exec_obj = new ze_sscp_executable_object{
        ctx, dev, hcf_object, compiled_image, config};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, kernel_name, backend_id::level_zero, hcf_object,
      code_object_selector, code_object_constructor);

  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"ze_queue: Code object construction failed"});
  }

  ze_kernel_handle_t kernel;
  result res = static_cast<const ze_executable_object *>(obj)->get_kernel(
      kernel_name, kernel);
  
  if(!res.is_success())
    return res;

  std::vector<ze_event_handle_t> wait_events =
      get_enqueued_event_handles();
  std::shared_ptr<dag_node_event> completion_evt = create_event();

  HIPSYCL_DEBUG_INFO << "ze_queue: Attempting to submit SSCP kernel"
                     << std::endl;


  glue::jit::cxx_argument_mapper arg_mapper{*kernel_info, args, arg_sizes,
                                            num_args};
  if(!arg_mapper.mapping_available()) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ze_queue: Could not map C++ arguments to kernel arguments"});
  }

  auto submission_err = submit_ze_kernel(
      kernel, get_ze_command_list(),
      static_cast<ze_node_event *>(completion_evt.get())->get_event_handle(),
      wait_events, group_size, num_groups, arg_mapper.get_mapped_args(),
      const_cast<std::size_t *>(arg_mapper.get_mapped_arg_sizes()),
      arg_mapper.get_mapped_num_args(), kernel_info);

  if(!submission_err.is_success())
    return submission_err;

  register_submitted_op(completion_evt);

  return make_success();
#else
  return make_error(
      __hipsycl_here(),
      error_info{"ze_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with Level Zero SSCP support."});
#endif
}


}
}

