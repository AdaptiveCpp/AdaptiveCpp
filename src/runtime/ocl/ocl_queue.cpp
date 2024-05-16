/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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


#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/executor.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/ocl/ocl_event.hpp"
#include "hipSYCL/runtime/ocl/ocl_queue.hpp"
#include "hipSYCL/runtime/ocl/ocl_hardware_manager.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/spirv/LLVMToSpirvFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include <CL/cl.h>

#endif

namespace hipsycl {
namespace rt {


namespace {

result submit_ocl_kernel(cl::Kernel& kernel,
                        cl::CommandQueue& queue,
                        const rt::range<3> &group_size,
                        const rt::range<3> &num_groups, void **kernel_args,
                        const std::size_t *arg_sizes, std::size_t num_args,
                        ocl_usm* usm,
                        const hcf_kernel_info *info,
                        cl::Event* evt_out = nullptr) {
  // All OpenCL API calls are safe, except calls that configure kernel objects
  // like clSetKernelArgs. Currently we are not guaranteed that each thread gets
  // its own separate kernel object, so we have to lock the submission process for now.
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock{mutex};

  cl_int err = 0;
  for(std::size_t i = 0; i < num_args; ++i ){
    HIPSYCL_DEBUG_INFO << "ocl_queue: Setting kernel argument " << i
                       << " of size " << arg_sizes[i] << " at " << kernel_args[i]
                       << std::endl;

    err = kernel.setArg(i, static_cast<std::size_t>(arg_sizes[i]), kernel_args[i]);

    if(err != CL_SUCCESS) {
      return make_error(
          __hipsycl_here(),
          error_info{"ocl_queue: Could not set kernel argument",
                     error_code{"CL", static_cast<int>(err)}});
    }
  }

  // This is necessary for USM pointers, which hipSYCL *always*
  // relies on.
  err = usm->enable_indirect_usm_access(kernel);

  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ocl_queue: Could not set indirect access flags",
                     error_code{"CL", static_cast<int>(err)}});
  }

  HIPSYCL_DEBUG_INFO << "ocl_queue: Submitting kernel!" << std::endl;
  rt::range<3> global_size = num_groups * group_size;
  
  cl::NDRange cl_global_size{global_size[0], global_size[1], global_size[2]};
  cl::NDRange cl_local_size{group_size[0], group_size[1], group_size[2]};
  cl::NDRange offset{0, 0, 0};
  if (global_size[2] == 1) {
    cl_global_size = cl::NDRange{global_size[0], global_size[1]};
    cl_local_size = cl::NDRange{group_size[0], group_size[1]};
    offset = cl::NDRange{0, 0};
    if (global_size[1] == 1) {
      cl_global_size = cl::NDRange{global_size[0]};
      cl_local_size = cl::NDRange{group_size[0]};
      offset = cl::NDRange{0};
    }
  }

  err = queue.enqueueNDRangeKernel(kernel, offset, cl_global_size,
                                   cl_local_size, nullptr, evt_out);

  if(err != CL_SUCCESS) {
    return make_error(
        __hipsycl_here(),
        error_info{"ocl_queue: Kernel launch failed",
                   error_code{"CL", static_cast<int>(err)}});
  }

  return make_success();
}

}

class ocl_hardware_manager;

ocl_queue::ocl_queue(ocl_hardware_manager* hw_manager, std::size_t device_index)
  : _hw_manager{hw_manager}, _device_index{device_index}, _sscp_invoker{this},
    _kernel_cache{kernel_cache::get()} {

  cl_command_queue_properties props = 0;
  ocl_hardware_context *dev_ctx =
      static_cast<ocl_hardware_context *>(hw_manager->get_device(device_index));
  cl::Device cl_dev = dev_ctx->get_cl_device();
  cl::Context cl_ctx = dev_ctx->get_cl_context();

  cl_int err;
  _queue = cl::CommandQueue{cl_ctx, cl_dev, props, &err};
  if(err != CL_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ocl_queue: Couldn't construct backend queue",
                              error_code{"CL", err}});
  }
}

ocl_queue::~ocl_queue() {}

std::shared_ptr<dag_node_event> ocl_queue::insert_event() {
  if(!_state.get_most_recent_event()) {
    // Normally, this code path should only be triggered
    // when no work has been submitted to the queue, and so
    // nothing needs to be synchronized with. Thus
    // the returned event should never actually be needed
    // by other nodes in the DAG.
    // However, if some work fails to execute, we can end up
    // with the situation that the "no work submitted yet" situation
    // appears at later stages in the program, when events
    // are expected to work correctly.
    // It is thus safer to enqueue a barrier here.
    cl::Event wait_evt;
    cl_int err = _queue.enqueueBarrierWithWaitList(nullptr, &wait_evt);

    if(err != CL_SUCCESS) {
      register_error(
            __hipsycl_here(),
            error_info{
                "ocl_queue: enqueueBarrierWithWaitList() failed",
                error_code{"CL", err}});
    }
    register_submitted_op(wait_evt);
    
  }

  return _state.get_most_recent_event();
}

std::shared_ptr<dag_node_event> ocl_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<cl::Event, ocl_node_event>>(
      this);
}

result ocl_queue::submit_memcpy(memcpy_operation &op, dag_node_ptr) {

  HIPSYCL_DEBUG_INFO << "ocl_queue: On device "
                     << _hw_manager->get_device_id(_device_index)
                     << ": Processing memcpy request from device "
                     << op.source().get_device() << " to "
                     << op.dest().get_device() << std::endl;

  // TODO We could probably unify some of the logic here between
  // backends

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

  cl::Event evt;

  if(dimension == 1) {
    ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
    ocl_usm* usm = ocl_ctx->get_usm_provider();
    cl_int err = usm->enqueue_memcpy(_queue, op.dest().get_access_ptr(),
                        op.source().get_access_ptr(),
                        op.get_num_transferred_bytes(), {}, &evt);

    if(err != CL_SUCCESS) {
      return make_error(
          __hipsycl_here(),
          error_info{"ocl_queue: enqueuing memcpy failed",
                     error_code{"CL", static_cast<int>(err)}});
    }
  } else {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ocl_queue: Multidimensional memory copies are not yet supported.",
            error_type::unimplemented});
  }

  register_submitted_op(evt);
  return make_success();
}

result ocl_queue::submit_kernel(kernel_operation &op, dag_node_ptr node) {

  rt::backend_kernel_launcher *l =
      op.get_launcher().find_launcher(backend_id::ocl);
  if (!l)
    return make_error(__hipsycl_here(),
                      error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_invoker);
  l->set_backend_capabilities(cap);
  
  // TODO: Instrumentation
  l->invoke(node.get(), op.get_launcher().get_kernel_configuration());

  return make_success();
}

result ocl_queue::submit_prefetch(prefetch_operation &op, dag_node_ptr) {
  ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
  ocl_usm* usm = ocl_ctx->get_usm_provider();

  cl::Event evt;
  cl_int err = 0;
  if(op.get_target().is_host()) {
    err = usm->enqueue_prefetch(_queue, op.get_pointer(), op.get_num_bytes(),
                                CL_MIGRATE_MEM_OBJECT_HOST, {}, &evt);
  } else {
    err = usm->enqueue_prefetch(_queue, op.get_pointer(), op.get_num_bytes(),
                                0, {}, &evt);
  }

  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ocl_queue: enqueuing prefetch failed",
                     error_code{"CL", static_cast<int>(err)}});
  }

  register_submitted_op(evt);
  return make_success();
}

result ocl_queue::submit_memset(memset_operation& op, dag_node_ptr) {
  ocl_hardware_context *ocl_ctx = static_cast<ocl_hardware_context *>(
        _hw_manager->get_device(_device_index));
  ocl_usm* usm = ocl_ctx->get_usm_provider();

  cl::Event evt;
  cl_int err = usm->enqueue_memset(_queue, op.get_pointer(), op.get_pattern(),
                                   op.get_num_bytes(), {}, &evt);
  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ocl_queue: enqueuing memset failed",
                     error_code{"CL", static_cast<int>(err)}});
  }

  register_submitted_op(evt);
  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result ocl_queue::submit_queue_wait_for(dag_node_ptr evt) {

  ocl_node_event *ocl_evt =
      static_cast<ocl_node_event *>(evt->get_event().get());
  
  std::vector<cl::Event> events{ocl_evt->get_event()};

  if (_hw_manager->get_context(ocl_evt->get_device()) !=
      _hw_manager->get_context(_hw_manager->get_device_id(_device_index))) {
    return submit_external_wait_for(evt);
  }

  cl::Event wait_evt;
  cl_int err = _queue.enqueueBarrierWithWaitList(&events, &wait_evt);

  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{
              "ocl_queue: enqueueBarrierWithWaitList() failed",
              error_code{"CL", err}});
  }
  register_submitted_op(wait_evt);
  return make_success();
}

result ocl_queue::submit_external_wait_for(dag_node_ptr node) {
  ocl_hardware_context* hw_ctx = static_cast<ocl_hardware_context *>(
      _hw_manager->get_device(_device_index));
  cl_int err;
  cl::UserEvent uevt{hw_ctx->get_cl_context(), &err};
  if(err != CL_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{
              "ocl_queue: OpenCL user event creation failed",
              error_code{"CL", err}});
  }

  cl::Event barrier_evt;
  std::vector<cl::Event> wait_events{uevt};
  _queue.enqueueBarrierWithWaitList(&wait_events, &barrier_evt);

  register_submitted_op(barrier_evt);
  _host_worker([uevt, node]() mutable{
    node->wait();
    cl_int err = uevt.setStatus(CL_COMPLETE);
    if(err != CL_SUCCESS) {
      register_error(
          __hipsycl_here(),
          error_info{"ocl_queue: Could not change status of user event",
                     error_code{"CL", err}});
    }
  });

  return make_success();
}

result ocl_queue::wait() {
  cl_int err = _queue.finish();
  if(err != CL_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"ocl_queue: Couldn't finish queue",
                                 error_code{"CL", err}});
  }
  return make_success();
}

device_id ocl_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}

/// Return native type if supported, nullptr otherwise
void* ocl_queue::get_native_type() const {
  return nullptr;
}

result ocl_queue::query_status(inorder_queue_status& status) {
  auto evt = _state.get_most_recent_event();
  if(evt) {
    status = inorder_queue_status{evt->is_complete()};
  } else {
    status = inorder_queue_status{true};
  }
  return make_success();
}

ocl_hardware_manager *ocl_queue::get_hardware_manager() const {
  return _hw_manager;
}

result ocl_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    const std::string &kernel_name, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const kernel_configuration &initial_config) {


#ifdef HIPSYCL_WITH_SSCP_COMPILER

  const hcf_kernel_info *kernel_info =
      rt::hcf_cache::get().get_kernel_info(hcf_object, kernel_name);
  if(!kernel_info) {
    return make_error(
        __hipsycl_here(),
        error_info{"ocl_queue: Could not obtain hcf kernel info for kernel " +
            kernel_name});
  }


  glue::jit::cxx_argument_mapper arg_mapper{*kernel_info, args, arg_sizes,
                                            num_args};
  if(!arg_mapper.mapping_available()) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ocl_queue: Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args, local_mem_size};

  ocl_hardware_context *hw_ctx = static_cast<ocl_hardware_context *>(
      _hw_manager->get_device(_device_index));
  cl::Context ctx = hw_ctx->get_cl_context();
  cl::Device dev = hw_ctx->get_cl_device();

  // Need to create custom config to ensure we can distinguish other
  // kernels compiled with different values e.g. of local mem allocation size
  static thread_local kernel_configuration config;
  config = initial_config;
  
  config.append_base_configuration(
      kernel_base_config_parameter::backend_id, backend_id::ocl);
  config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow,
      compilation_flow::sscp);
  config.append_base_configuration(
      kernel_base_config_parameter::hcf_object_id, hcf_object);
  
  for(const auto& flag : kernel_info->get_compilation_flags())
    config.set_build_flag(flag);
  for(const auto& opt : kernel_info->get_compilation_options())
    config.set_build_option(opt.first, opt.second);

  config.set_build_option(
      kernel_build_option::spirv_dynamic_local_mem_allocation_size,
      local_mem_size);

  // TODO: Enable this if we are on Intel
  // config.set_build_flag(kernel_build_flag::spirv_enable_intel_llvm_spirv_options);

  auto binary_configuration_id = adaptivity_engine.finalize_binary_configuration(config);
  auto code_object_configuration_id = binary_configuration_id;
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_device, dev.get());
  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_context, ctx.get());

 

  
  auto jit_compiler = [&](std::string& compiled_image) -> bool {
    const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        adaptivity_engine.select_image_and_kernels(&kernel_names);

    // Construct SPIR-V translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      std::move(compiler::createLLVMToSpirvTranslator(kernel_names));
    
    // Lower kernels to SPIR-V
    auto err = glue::jit::compile(translator.get(),
        hcf, selected_image_name, config, compiled_image);
    
    if(!err.is_success()) {
      register_error(err);
      return false;
    }
    return true;
  };

  auto code_object_constructor = [&](const std::string& compiled_image) -> code_object* {
    ocl_executable_object *exec_obj = new ocl_executable_object{
        ctx, dev, hcf_object, compiled_image, config};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_jit_code_object(
      code_object_configuration_id, binary_configuration_id,
      jit_compiler, code_object_constructor);

  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"ocl_queue: Code object construction failed"});
  }

  cl::Kernel kernel;
  result res = static_cast<const ocl_executable_object *>(obj)->get_kernel(
      kernel_name, kernel);
  
  if(!res.is_success())
    return res;

  HIPSYCL_DEBUG_INFO << "ocl_queue: Attempting to submit SSCP kernel"
                     << std::endl;

  cl::Event completion_evt;
  auto submission_err = submit_ocl_kernel(
      kernel, _queue, group_size, num_groups, arg_mapper.get_mapped_args(),
      const_cast<std::size_t *>(arg_mapper.get_mapped_arg_sizes()),
      arg_mapper.get_mapped_num_args(), hw_ctx->get_usm_provider(), kernel_info,
      &completion_evt);

  if(!submission_err.is_success())
    return submission_err;

  register_submitted_op(completion_evt);

  return make_success();
#else
  return make_error(
      __hipsycl_here(),
      error_info{"ocl_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with OpenCL SSCP support."});
#endif

}


void ocl_queue::register_submitted_op(cl::Event evt) {
  this->_state.set_most_recent_event(std::make_shared<ocl_node_event>(
      _hw_manager->get_device_id(_device_index), evt));
}

}
}

