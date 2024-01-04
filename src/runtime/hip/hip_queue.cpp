/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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

#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/hip/hip_hardware_manager.hpp"
#include "hipSYCL/runtime/hip/hip_queue.hpp"
#include "hipSYCL/runtime/hip/hip_backend.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/hip/hip_code_object.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/amdgpu/LLVMToAmdgpuFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"

#endif

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

void host_synchronization_callback(hipStream_t stream, hipError_t status,
                                   void *userData) {
  
  assert(userData);
  dag_node_ptr* node = static_cast<dag_node_ptr*>(userData);
  
  if(status != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_queue callback: HIP returned error code.",
                              error_code{"HIP", status}});
  }
  else {
    (*node)->wait();
  }
  delete node;
}



class hip_instrumentation_guard {
public:
  hip_instrumentation_guard(hip_queue *q,
                             operation &op, dag_node_ptr node) 
                             : _queue{q}, _operation{&op}, _node{node} {
    assert(q);
    
    if(!_node)
      return;

    if (_node->get_execution_hints()
            .has_hint<
                rt::hints::request_instrumentation_submission_timestamp>()) {

      op.get_instrumentations()
          .add_instrumentation<instrumentations::submission_timestamp>(
            std::make_shared<hip_submission_timestamp>(profiler_clock::now()));
    }

    if (_node->get_execution_hints().has_hint<
                rt::hints::request_instrumentation_start_timestamp>()) {

      _task_start = _queue->insert_event();

      op.get_instrumentations()
          .add_instrumentation<instrumentations::execution_start_timestamp>(
              std::make_shared<hip_execution_start_timestamp>(
                  _queue->get_timing_reference(), _task_start));
    }
  }

  ~hip_instrumentation_guard() {
    if(!_node)
      return;

    if (_node->get_execution_hints()
            .has_hint<rt::hints::request_instrumentation_finish_timestamp>()) {
      std::shared_ptr<dag_node_event> task_finish = _queue->insert_event();

      if(_task_start) {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<hip_execution_finish_timestamp>(
                    _queue->get_timing_reference(), _task_start, task_finish));
      } else {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<hip_execution_finish_timestamp>(
                    _queue->get_timing_reference(), task_finish));
      }
    }
  }

private:
  hip_queue* _queue;
  operation* _operation;
  dag_node_ptr _node;
  std::shared_ptr<dag_node_event> _task_start;
};

result launch_kernel_from_module(ihipModule_t *module,
                                 const std::string &kernel_name,
                                 const rt::range<3> &grid_size,
                                 const rt::range<3> &block_size,
                                 unsigned dynamic_shared_mem,
                                 hipStream_t stream, void **kernel_args,
                                 std::size_t *arg_sizes, std::size_t num_args) {

  hipFunction_t kernel_func;
  hipError_t err =
      hipModuleGetFunction(&kernel_func, module, kernel_name.c_str());

  if(err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: could not extract kernel from module",
                                 error_code{"HIP", static_cast<int>(err)}});
  }

  err = hipModuleLaunchKernel(kernel_func, 
    static_cast<unsigned>(grid_size.get(0)),
                       static_cast<unsigned>(grid_size.get(1)),
                       static_cast<unsigned>(grid_size.get(2)),
                       static_cast<unsigned>(block_size.get(0)),
                       static_cast<unsigned>(block_size.get(1)),
                       static_cast<unsigned>(block_size.get(2)),
                       dynamic_shared_mem, stream, kernel_args, nullptr);

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: could not submit kernel from module",
                                 error_code{"HIP", static_cast<int>(err)}});
  }
  

  return make_success();
}
}


void hip_queue::activate_device() const {
  hip_device_manager::get().activate_device(_dev.get_id());
}

hip_queue::hip_queue(hip_backend *be, device_id dev, int priority)
    : _dev{dev}, _stream{nullptr}, _backend{be},
      _multipass_code_object_invoker{this}, _sscp_code_object_invoker{this} {
  this->activate_device();

  hipError_t err;
  if(priority == 0) {
    err = hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking);
  } else {
    // hipStreamCreateWithPriority will clamp priority to the allowed range
    err = hipStreamCreateWithPriority(&_stream, hipStreamNonBlocking, priority);
  }
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_queue: Couldn't construct backend stream",
                              error_code{"HIP", err}});
    return;
  }

  _reference_event = host_timestamped_event{this};
}

hipStream_t hip_queue::get_stream() const { return _stream; }

hip_queue::~hip_queue() {
  auto err = hipStreamDestroy(_stream);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_queue: Couldn't destroy stream",
                              error_code{"HIP", err}});
  }
}

/// Inserts an event into the stream
std::shared_ptr<dag_node_event> hip_queue::insert_event() {
  hipEvent_t evt;
  auto event_creation_result =
      _backend->get_event_pool(_dev)->obtain_event(evt);
  if(!event_creation_result.is_success()) {
    register_error(event_creation_result);
    return nullptr;
  }

  hipError_t err = hipEventRecord(evt, this->get_stream());
  if (err != hipSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"hip_queue: Couldn't record event", error_code{"HIP", err}});
    return nullptr;
  }

  return std::make_shared<hip_node_event>(_dev, std::move(evt),
                                          _backend->get_event_pool(_dev));
}

std::shared_ptr<dag_node_event> hip_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<hipEvent_t, hip_node_event>>(
      this);
}



result hip_queue::submit_memcpy(memcpy_operation & op, dag_node_ptr node) {

  device_id source_dev = op.source().get_device();
  device_id dest_dev = op.dest().get_device();

  assert(op.source().get_access_ptr());
  assert(op.dest().get_access_ptr());

  hipMemcpyKind copy_kind = hipMemcpyHostToDevice;

  if (source_dev.get_full_backend_descriptor().sw_platform == api_platform::hip) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::hip) {
      assert(source_dev.get_full_backend_descriptor().hw_platform ==
                 dest_dev.get_full_backend_descriptor().hw_platform &&
             "Attempted to execute explicit device<->device copy operation "
             "between devices from different HIP hardware backends");
      copy_kind = hipMemcpyDeviceToDevice;

    } else if (dest_dev.get_full_backend_descriptor().hw_platform ==
               hardware_platform::cpu) {
      copy_kind = hipMemcpyDeviceToHost;
      
    } else
      assert(false && "Unknown copy destination platform");
  } else if (source_dev.get_full_backend_descriptor().hw_platform ==
             hardware_platform::cpu) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::hip) {
      copy_kind = hipMemcpyHostToDevice;
    } else
      assert(false && "Unknown copy destination platform");
  } else
    assert(false && "Unknown copy source platform");


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

  hip_instrumentation_guard instrumentation{this, op, node};

  hipError_t err = hipSuccess;
  if (dimension == 1) {

    err = hipMemcpyAsync(
        op.dest().get_access_ptr(), op.source().get_access_ptr(),
        op.get_num_transferred_bytes(), copy_kind, get_stream());
    
  } else if (dimension == 2) {
    err = hipMemcpy2DAsync(
        op.dest().get_access_ptr(),
        extract_from_range3<2>(op.dest().get_allocation_shape())[1] *
            op.dest().get_element_size(),
        op.source().get_access_ptr(),
        extract_from_range3<2>(op.source().get_allocation_shape())[1] *
            op.source().get_element_size(),
        extract_from_range3<2>(op.get_num_transferred_elements())[1] *
            op.source().get_element_size(),
        extract_from_range3<2>(op.get_num_transferred_elements())[0], copy_kind,
        get_stream());
  } else {
    hipMemcpy3DParms params = {0};
    params.srcPtr = make_hipPitchedPtr(op.source().get_access_ptr(),
                                       op.source().get_allocation_shape()[2] *
                                           op.source().get_element_size(),
                                       op.source().get_allocation_shape()[2],
                                       op.source().get_allocation_shape()[1]);
    params.dstPtr = make_hipPitchedPtr(op.dest().get_access_ptr(),
                                       op.dest().get_allocation_shape()[2] *
                                           op.dest().get_element_size(),
                                       op.dest().get_allocation_shape()[2],
                                       op.dest().get_allocation_shape()[1]);
    params.extent = {op.get_num_transferred_elements()[2] *
                         op.source().get_element_size(),
                     op.get_num_transferred_elements()[1],
                     op.get_num_transferred_elements()[0]};
    params.kind = copy_kind;

    err = hipMemcpy3DAsync(&params, get_stream());
  }

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Couldn't submit memcpy",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

result hip_queue::submit_kernel(kernel_operation &op, dag_node_ptr node) {

  this->activate_device();
  rt::backend_kernel_launcher *l =
      op.get_launcher().find_launcher(backend_id::hip);
  
  if (!l)
    return make_error(__hipsycl_here(), error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);
  
  rt::backend_kernel_launch_capabilities cap;
  
  cap.provide_multipass_invoker(&_multipass_code_object_invoker);
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);

  l->set_backend_capabilities(cap);

  hip_instrumentation_guard instrumentation{this, op, node};
  l->invoke(node.get(), op.get_launcher().get_kernel_configuration());

  return make_success();
}

result hip_queue::submit_prefetch(prefetch_operation& op, dag_node_ptr node) {
  // Need to enable instrumentation even if we cannot enable actual
  // prefetches so that the user will be able to access instrumentation
  // properties of the event.
  hip_instrumentation_guard instrumentation{this, op, node};
#ifdef HIPSYCL_RT_HIP_SUPPORTS_UNIFIED_MEMORY
  
  hipError_t err = hipSuccess;
  
  if (op.get_target().is_host()) {
    err = hipMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                              hipCpuDeviceId, get_stream());
  } else {
    err = hipMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                              _dev.get_id(), get_stream());
  }

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: hipMemPrefetchAsync() failed",
                                 error_code{"HIP", err}});
  }
#else
  // HIP does not yet support prefetching functions
  HIPSYCL_DEBUG_WARNING << "Ignoring prefetch request because HIP does not yet "
                           "support prefetching memory."
                        << std::endl;

#endif
  return make_success();
}

result hip_queue::submit_memset(memset_operation &op, dag_node_ptr node) {

  hip_instrumentation_guard instrumentation{this, op, node};
  hipError_t err = hipMemsetAsync(op.get_pointer(), op.get_pattern(),
                                  op.get_num_bytes(), get_stream());

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: hipMemsetAsync() failed",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

result hip_queue::wait() {

  auto err = hipStreamSynchronize(_stream);

  if(err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Couldn't synchronize with stream",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

result hip_queue::query_status(inorder_queue_status &status) {
  auto err = hipStreamQuery(_stream);
  if(err == hipSuccess) {
    status = inorder_queue_status{true};
  } else if(err == hipErrorNotReady) {
    status = inorder_queue_status{false};
  } else {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Could not query stream status",
                                 error_code{"HIP", static_cast<int>(err)}});
  }

  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result hip_queue::submit_queue_wait_for(dag_node_ptr node) {
  auto evt = node->get_event();
  assert(dynamic_is<inorder_queue_event<hipEvent_t>>(evt.get()));

  inorder_queue_event<hipEvent_t> *hip_evt =
      cast<inorder_queue_event<hipEvent_t>>(evt.get());
  auto err = hipStreamWaitEvent(_stream, hip_evt->request_backend_event(), 0);
  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: hipStreamWaitEvent() failed",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

result hip_queue::submit_external_wait_for(dag_node_ptr node) {

  dag_node_ptr* user_data = new dag_node_ptr;
  assert(user_data);
  *user_data = node;

  auto err = 
      hipStreamAddCallback(_stream, host_synchronization_callback,
                           reinterpret_cast<void *>(user_data), 0);

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                   error_info{"hip_queue: Couldn't submit stream callback",
                              error_code{"HIP", err}});
  }
  
  return make_success();
}

device_id hip_queue::get_device() const { return _dev; }

void *hip_queue::get_native_type() const {
  return static_cast<void*>(get_stream());
}

result hip_queue::submit_multipass_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &backend_kernel_name, const rt::range<3> &grid_size,
      const rt::range<3> &block_size, unsigned dynamic_shared_mem,
      void **kernel_args, std::size_t* arg_sizes, std::size_t num_args) {

  this->activate_device();
  
  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t *kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"hip_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  const common::hcf_container *hcf =
        rt::hcf_cache::get().get_hcf(hcf_object);
  if (!hcf)
    return make_error(
        __hipsycl_here(),
        error_info{"hip_queue: Could not access requested HCF object"});

  assert(hcf->root_node());
  std::vector<std::string> available_targets = hcf->root_node()->get_subnodes();
  assert(!available_targets.empty());

  // TODO Select correct target based on actual device - currently
  // we just use the first device image no matter which device it was
  // compiled for
  std::string selected_target = available_targets[0];
  int device = _dev.get_id();

  auto code_object_selector = [&](const code_object* candidate) -> bool {
    // Also no need to check for HIP backend since the kernel cache already
    // guarantees that we are only given candidates for the requested backend (CUDA).
    return (candidate->target_arch() == selected_target) &&
           (candidate->state() == code_object_state::executable) &&
           (static_cast<const hip_executable_object *>(candidate)
                ->source_compilation_flow() ==
            compilation_flow::explicit_multipass) &&
           (static_cast<const hip_executable_object *>(candidate)
                ->get_device() == device);
  };

  // Will be invoked by the kernel cache in case there is a miss in the kernel
  // cache and we have to construct a new code object

  auto code_object_constructor = [&]() -> code_object * {

    const common::hcf_container::node* target_node =
      hcf->root_node()->get_subnode(selected_target);
    if(!target_node)
      return nullptr;
    if(!target_node->has_binary_data_attached())
      return nullptr;

    std::string kernel_image;
    if(!hcf->get_binary_attachment(target_node, kernel_image)){
      HIPSYCL_DEBUG_ERROR
          << "hip_queue: Could not extract hip fat binary code from "
             "HCF node; invalid HCF data?"
          << std::endl;
      return nullptr;
    }

    hip_executable_object *exec_obj = new hip_multipass_executable_object{
        hcf_object, selected_target, kernel_image, device};
    result r = exec_obj->get_build_result();

    if (!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, backend_kernel_name, backend_id::hip, hcf_object,
      code_object_selector, code_object_constructor);

  
  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Code object construction failed"});
  }

  // Since we only support explicit multipass in HIP with clang >= 13 and therefore
  // are in the new kernel name mangling logic path, we can assume that we know
  // the full kernel name, and don't need to query available kernels in the device image
  // as in the CUDA backend.
  return launch_kernel_from_module(
      static_cast<const hip_executable_object *>(obj)->get_module(),
      backend_kernel_name, grid_size, block_size, dynamic_shared_mem, _stream,
      kernel_args, arg_sizes, num_args);
}

result hip_queue::submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &kernel_name, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const glue::kernel_configuration &config) {
#ifdef HIPSYCL_WITH_SSCP_COMPILER
  this->activate_device();
  
  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t *kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"hip_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  auto configuration_id = config.generate_id();
  int device = _dev.get_id();

  hip_hardware_context *ctx = static_cast<hip_hardware_context *>(
      this->_backend->get_hardware_manager()->get_device(device));

  std::string target_arch_name = ctx->get_device_arch();

  const hcf_kernel_info *kernel_info =
      rt::hcf_cache::get().get_kernel_info(hcf_object, kernel_name);
  if(!kernel_info) {
    return make_error(
        __hipsycl_here(),
        error_info{"hip_queue: Could not obtain hcf kernel info for kernel " +
            global_kernel_name});
  }

  auto code_object_selector = [&](const code_object* candidate) -> bool {
    
    if ((candidate->managing_backend() != backend_id::hip) ||
        (candidate->source_compilation_flow() != compilation_flow::sscp) ||
        (candidate->state() != code_object_state::executable))
      return false;

    const hip_sscp_executable_object *obj =
        static_cast<const hip_sscp_executable_object *>(candidate);
    
    if(obj->configuration_id() != configuration_id)
      return false;

    return obj->get_device() == device;
  };

  auto code_object_constructor = [&]() -> code_object * {
    const common::hcf_container *hcf =
        rt::hcf_cache::get().get_hcf(hcf_object);
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        glue::jit::select_image(kernel_info, &kernel_names);

    // Construct amdgpu translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      compiler::createLLVMToAmdgpuTranslator(kernel_names);

    translator->setBuildOption("amdgpu-target-device", target_arch_name);

    // Lower kernels
    std::string amdgpu_image;
    auto err = glue::jit::compile(translator.get(),
        hcf, selected_image_name, config, amdgpu_image);
    
    if(!err.is_success()) {
      register_error(err);
      return nullptr;
    }

    hip_sscp_executable_object *exec_obj = new hip_sscp_executable_object{
        amdgpu_image, target_arch_name, hcf_object,
        kernel_names, device,           config};
    result r = exec_obj->get_build_result();

    HIPSYCL_DEBUG_INFO
        << "hip_queue: Successfully compiled SSCP kernels to module " << exec_obj->get_module()
        << std::endl;

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, kernel_name, backend_id::hip, hcf_object,
      code_object_selector, code_object_constructor);

  
  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Code object construction failed"});
  }

  ihipModule_t *module =
      static_cast<const hip_executable_object *>(obj)->get_module();
  assert(module);

  glue::jit::cxx_argument_mapper arg_mapper{*kernel_info, args, arg_sizes,
                                            num_args};
  if(!arg_mapper.mapping_available()) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "hip_queue: Could not map C++ arguments to kernel arguments"});
  }

  return launch_kernel_from_module(
      module, kernel_name, num_groups, group_size, local_mem_size, _stream,
      arg_mapper.get_mapped_args(),
      const_cast<std::size_t *>(arg_mapper.get_mapped_arg_sizes()),
      arg_mapper.get_mapped_num_args());
#else
  return make_error(
      __hipsycl_here(),
      error_info{
          "hip_queue: SSCP kernel launch was requested, but hipSYCL was "
          "not built with HIP SSCP support."});
#endif
}

hip_multipass_code_object_invoker::hip_multipass_code_object_invoker(
    hip_queue *q)
    : _queue{q} {}

result hip_multipass_code_object_invoker::submit_kernel(
    const kernel_operation& op,
    hcf_object_id hcf_object,
    const rt::range<3> &num_groups,
    const rt::range<3> &group_size,
    unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const std::string &kernel_name_tag,
    const std::string &kernel_body_name) {

  assert(_queue);

  std::string kernel_name = kernel_body_name;
  if(kernel_name_tag.find("__hipsycl_unnamed_kernel") == std::string::npos)
    kernel_name = kernel_name_tag;

  return _queue->submit_multipass_kernel_from_code_object(
      op, hcf_object, kernel_name, num_groups, group_size, local_mem_size, args,
      arg_sizes, num_args);
}

result hip_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const std::string &kernel_name,
    const glue::kernel_configuration &config) {

  return _queue->submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, num_groups, group_size, local_mem_size, args,
      arg_sizes, num_args, config);
}

}
}

