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

#include "driver_types.h"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/cuda/cuda_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/cuda/cuda_queue.hpp"
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/cuda/cuda_code_object.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/inorder_queue.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER

#include "hipSYCL/compiler/llvm-to-backend/ptx/LLVMToPtxFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"

#endif

#include <cuda_runtime_api.h>
#include <cuda_runtime.h> //for make_cudaPitchedPtr
#include <cuda.h> // For kernels launched from modules

#include <cassert>
#include <memory>

namespace hipsycl {
namespace rt {

namespace {

void host_synchronization_callback(cudaStream_t stream, cudaError_t status,
                                   void *userData) {
  
  assert(userData);
  dag_node_ptr* node = static_cast<dag_node_ptr*>(userData);
  
  if(status != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_queue callback: CUDA returned error code.",
                              error_code{"CUDA", status}});
  }
  else {
    (*node)->wait();
  }
  delete node;
}


class cuda_instrumentation_guard {
public:
  cuda_instrumentation_guard(cuda_queue *q,
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
            std::make_shared<cuda_submission_timestamp>(profiler_clock::now()));
    }

    if (_node->get_execution_hints().has_hint<
                rt::hints::request_instrumentation_start_timestamp>()) {

      _task_start = _queue->insert_event();

      op.get_instrumentations()
          .add_instrumentation<instrumentations::execution_start_timestamp>(
              std::make_shared<cuda_execution_start_timestamp>(
                  _queue->get_timing_reference(), _task_start));
    }
  }

  ~cuda_instrumentation_guard() {
    if(!_node)
      return;
    
    if (_node->get_execution_hints()
            .has_hint<rt::hints::request_instrumentation_finish_timestamp>()) {
      std::shared_ptr<dag_node_event> task_finish = _queue->insert_event();

      if(_task_start) {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<cuda_execution_finish_timestamp>(
                    _queue->get_timing_reference(), _task_start, task_finish));
      } else {
        _operation->get_instrumentations()
            .add_instrumentation<instrumentations::execution_finish_timestamp>(
                std::make_shared<cuda_execution_finish_timestamp>(
                    _queue->get_timing_reference(), task_finish));
      }
    }
  }

private:
  cuda_queue* _queue;
  operation* _operation;
  dag_node_ptr _node;
  std::shared_ptr<dag_node_event> _task_start;
};

result launch_kernel_from_module(CUmodule module,
                                 const std::string &kernel_name,
                                 const rt::range<3> &grid_size,
                                 const rt::range<3> &block_size,
                                 unsigned shared_memory, cudaStream_t stream,
                                 void **kernel_args) {
  CUfunction f;
  CUresult err = cuModuleGetFunction(&f, module, kernel_name.c_str());

  if (err != CUDA_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: could not extract kernel from module",
                                 error_code{"CU", static_cast<int>(err)}});
  }

  err = cuLaunchKernel(f, static_cast<unsigned>(grid_size.get(0)),
                       static_cast<unsigned>(grid_size.get(1)),
                       static_cast<unsigned>(grid_size.get(2)),
                       static_cast<unsigned>(block_size.get(0)),
                       static_cast<unsigned>(block_size.get(1)),
                       static_cast<unsigned>(block_size.get(2)),
                       shared_memory, stream, kernel_args, nullptr);

  if (err != CUDA_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: could not submit kernel from module",
                                 error_code{"CU", static_cast<int>(err)}});
  }
  
  return make_success();
}
}


void cuda_queue::activate_device() const {
  cuda_device_manager::get().activate_device(_dev.get_id());
}

cuda_queue::cuda_queue(cuda_backend *be, device_id dev, int priority)
    : _dev{dev}, _multipass_code_object_invoker{this},
      _sscp_code_object_invoker{this}, _stream{nullptr}, _backend{be} {
  this->activate_device();

  cudaError_t err;
  if(priority == 0) {
    err = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
  } else {
    // CUDA API will clamp the priority to the range 
    err = cudaStreamCreateWithPriority(&_stream, cudaStreamNonBlocking, priority);
  }
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_queue: Couldn't construct backend stream",
                              error_code{"CUDA", err}});
    return;
  }

  _reference_event = host_timestamped_event{this};
}

CUstream_st* cuda_queue::get_stream() const { return _stream; }

cuda_queue::~cuda_queue() {
  auto err = cudaStreamDestroy(_stream);
  if (err != cudaSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"cuda_queue: Couldn't destroy stream",
                              error_code{"CUDA", err}});
  }
}

/// Inserts an event into the stream
std::shared_ptr<dag_node_event> cuda_queue::insert_event() {
  cudaEvent_t evt;
  auto event_creation_result =
      _backend->get_event_pool(_dev)->obtain_event(evt);
  if(!event_creation_result.is_success()) {
    register_error(event_creation_result);
    return nullptr;
  }

  cudaError_t err = cudaEventRecord(evt, this->get_stream());
  if (err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Couldn't record event", error_code{"CUDA", err}});
    return nullptr;
  }

  return std::make_shared<cuda_node_event>(_dev, evt,
                                           _backend->get_event_pool(_dev));
}

std::shared_ptr<dag_node_event> cuda_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<cudaEvent_t, cuda_node_event>>(
      this);
}


result cuda_queue::submit_memcpy(memcpy_operation & op, dag_node_ptr node) {

  device_id source_dev = op.source().get_device();
  device_id dest_dev = op.dest().get_device();

  assert(op.source().get_access_ptr());
  assert(op.dest().get_access_ptr());

  cudaMemcpyKind copy_kind = cudaMemcpyHostToDevice;

  if (source_dev.get_full_backend_descriptor().sw_platform == api_platform::cuda) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::cuda) {
      assert(source_dev.get_full_backend_descriptor().hw_platform ==
                 dest_dev.get_full_backend_descriptor().hw_platform &&
             "Attempted to execute explicit device<->device copy operation "
             "between devices from different CUDA hardware backends");
      copy_kind = cudaMemcpyDeviceToDevice;
    } else if (dest_dev.get_full_backend_descriptor().hw_platform ==
               hardware_platform::cpu) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else
      assert(false && "Unknown copy destination platform");
  } else if (source_dev.get_full_backend_descriptor().hw_platform ==
             hardware_platform::cpu) {
    if (dest_dev.get_full_backend_descriptor().sw_platform ==
        api_platform::cuda) {
      copy_kind = cudaMemcpyHostToDevice;
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


  cuda_instrumentation_guard instrumentation{this, op, node};

  cudaError_t err = cudaSuccess;
  if (dimension == 1) {
    err = cudaMemcpyAsync(
        op.dest().get_access_ptr(), op.source().get_access_ptr(),
        op.get_num_transferred_bytes(), copy_kind, get_stream());
    
  } else if (dimension == 2) {
    err = cudaMemcpy2DAsync(
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
    
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(op.source().get_access_ptr(),
                                        op.source().get_allocation_shape()[2] *
                                            op.source().get_element_size(),
                                        op.source().get_allocation_shape()[2],
                                        op.source().get_allocation_shape()[1]);
    params.dstPtr = make_cudaPitchedPtr(op.dest().get_access_ptr(),
                                        op.dest().get_allocation_shape()[2] *
                                            op.dest().get_element_size(),
                                        op.dest().get_allocation_shape()[2],
                                        op.dest().get_allocation_shape()[1]);
    params.extent = {op.get_num_transferred_elements()[2] *
                        op.source().get_element_size(),
                    op.get_num_transferred_elements()[1],
                    op.get_num_transferred_elements()[0]};
    params.kind = copy_kind;

    err = cudaMemcpy3DAsync(&params, get_stream());
  }

  if (err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Couldn't submit memcpy",
                                  error_code{"CUDA", err}});
  }
  
  return make_success();
}

result cuda_queue::submit_kernel(kernel_operation &op, dag_node_ptr node) {

  this->activate_device();
  rt::backend_kernel_launcher *l =
      op.get_launcher().find_launcher(backend_id::cuda);
  if (!l)
    return make_error(__hipsycl_here(), error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);

  rt::backend_kernel_launch_capabilities cap;
  cap.provide_multipass_invoker(&_multipass_code_object_invoker);
  cap.provide_sscp_invoker(&_sscp_code_object_invoker);
  l->set_backend_capabilities(cap);
  
  cuda_instrumentation_guard instrumentation{this, op, node};
  l->invoke(node.get(), op.get_launcher().get_kernel_configuration());

  return make_success();
}

result cuda_queue::submit_prefetch(prefetch_operation& op, dag_node_ptr node) {
#ifndef _WIN32
  
  cudaError_t err = cudaSuccess;
  
  cuda_instrumentation_guard instrumentation{this, op, node};
  if (op.get_target().is_host()) {
    err = cudaMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                                        cudaCpuDeviceId, get_stream());
  } else {
    err = cudaMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                                        _dev.get_id(), get_stream());
  }


  if (err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: cudaMemPrefetchAsync() failed",
                                 error_code{"CUDA", err}});
  }
#else
  HIPSYCL_DEBUG_WARNING << "cuda_queue: Ignoring prefetch() hint"
                        << std::endl;
#endif // _WIN32
  return make_success();
}

result cuda_queue::submit_memset(memset_operation &op, dag_node_ptr node) {

  cuda_instrumentation_guard instrumentation{this, op, node};
  
  cudaError_t err = cudaMemsetAsync(op.get_pointer(), op.get_pattern(),
                                    op.get_num_bytes(), get_stream());
  

  if (err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: cudaMemsetAsync() failed",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result cuda_queue::submit_queue_wait_for(dag_node_ptr node) {
  auto evt = node->get_event();
  assert(dynamic_is<inorder_queue_event<cudaEvent_t>>(evt.get()));

  inorder_queue_event<cudaEvent_t> *cuda_evt =
      cast<inorder_queue_event<cudaEvent_t>>(evt.get());
  
  auto err = cudaStreamWaitEvent(_stream, cuda_evt->request_backend_event(), 0);
  if (err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: cudaStreamWaitEvent() failed",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

result cuda_queue::submit_external_wait_for(dag_node_ptr node) {

  dag_node_ptr* user_data = new dag_node_ptr;
  assert(user_data);
  *user_data = node;

  auto err = 
      cudaStreamAddCallback(_stream, host_synchronization_callback,
                           reinterpret_cast<void *>(user_data), 0);

  if (err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Couldn't submit stream callback",
                                 error_code{"CUDA", err}});
  }
  
  return make_success();
}

result cuda_queue::wait() {

  auto err = cudaStreamSynchronize(_stream);

  if(err != cudaSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Couldn't synchronize with stream",
                                 error_code{"CUDA", err}});
  }

  return make_success();
}

result cuda_queue::submit_multipass_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    const std::string &backend_kernel_name, const rt::range<3> &grid_size,
    const rt::range<3> &block_size, unsigned dynamic_shared_mem,
    void **kernel_args, std::size_t num_args) {

  this->activate_device();

  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t *kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  // For now we need to extract HCF in order to get a list of available
  // compilation targets (list of embedded device images in HCF).
  // TODO we could cache this vector to avoid retrieving HCF for every kernel launch
  const common::hcf_container *hcf =
        rt::hcf_cache::get().get_hcf(hcf_object);
  if (!hcf)
    return make_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Could not access requested HCF object"});

  assert(hcf->root_node());
  std::vector<std::string> available_targets = hcf->root_node()->get_subnodes();
  assert(!available_targets.empty());

  // TODO Select best compiled target based on actual device - currently
  // we just use the first device image no matter which device it was
  // compiled for
  std::string selected_target = available_targets[0];

  int device = _dev.get_id();

  // This defines the conditions that we apply when looking for appropriate
  // code objects.
  // The correct hcf id and backend are already enforced by the kernel cache,
  // so we don't need to verify here.
  auto code_object_selector = [&](const code_object* candidate) -> bool {
    // We do not need to check for ptx since all CUDA code objects are PTX
    // Also no need to check for CUDA backend since the kernel cache already
    // guarantees that we are only given candidates for the requested backend (CUDA).
    return (candidate->target_arch() == selected_target) &&
           (candidate->state() == code_object_state::executable) &&
           (static_cast<const cuda_executable_object *>(candidate)
                ->get_device() == device) &&
           (candidate->source_compilation_flow() ==
            compilation_flow::explicit_multipass);
  };

  // Will be invoked by the kernel cache in case there is a miss in the kernel
  // cache and we have to construct a new code object
  auto code_object_constructor = [&]() -> code_object* {

    // First we need to obtain the source object. Need to use recursive_*
    // to avoid deadlocks since we are already inside the kernel cache here.

    auto source_object_selector = [&](const code_object *candidate) -> bool {
      return (candidate->state() == code_object_state::source) &&
             (candidate->target_arch() == selected_target);
    };

    auto source_object_constructor = [&]() -> code_object* {
      const common::hcf_container::node *tn =
          hcf->root_node()->get_subnode(selected_target);
      if(!tn)
        return nullptr;
      if(!tn->has_binary_data_attached())
        return nullptr;
      
      std::string source_code;
      if(!hcf->get_binary_attachment(tn, source_code)) {
        HIPSYCL_DEBUG_ERROR << "cuda_queue: Could not extract PTX code from "
                               "HCF node; invalid HCF data?"
                            << std::endl;
        return nullptr;
      }
      // The source object parses the PTX code and can give out information
      // such as a list of contained kernels
      return new cuda_source_object{hcf_object, selected_target, source_code};
    };

    const cuda_source_object *source = static_cast<const cuda_source_object *>(
        rt::kernel_cache::get().recursive_get_or_construct_code_object(
            *kidx, backend_kernel_name, backend_id::cuda, hcf_object,
            source_object_selector, source_object_constructor));

    cuda_executable_object *exec_obj =
        new cuda_multipass_executable_object{source, device};
    result r = exec_obj->get_build_result();

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, backend_kernel_name, backend_id::cuda, hcf_object,
      code_object_selector, code_object_constructor);

  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Code object construction failed"});
  }

  CUmodule cumodule = static_cast<const cuda_executable_object*>(obj)->get_module();
  assert(cumodule);

  // Need to find out full backend kernel name. This is necessary because
  // we don't know the *exact* kernel name until we know that we are in the clang 13+
  // name mangling path. It can be that we only have a fragment :(
  std::string full_kernel_name;
  for(const auto& name : obj->supported_backend_kernel_names()) {
    if(name.find(backend_kernel_name) != std::string::npos) {
      full_kernel_name = name;
      break;
    }
  }
  if(full_kernel_name.empty())
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Could not discover full kernel "
                                 "name from partial backend kernel name"});

  return launch_kernel_from_module(cumodule, full_kernel_name, grid_size,
                                   block_size, dynamic_shared_mem, _stream,
                                   kernel_args);
}


result cuda_queue::submit_sscp_kernel_from_code_object(
    const kernel_operation &op, hcf_object_id hcf_object,
    const std::string &kernel_name, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const glue::kernel_configuration &config) {
#ifdef HIPSYCL_WITH_SSCP_COMPILER

  this->activate_device();

  std::string global_kernel_name = op.get_global_kernel_name();
  const kernel_cache::kernel_name_index_t* kidx =
      kernel_cache::get().get_global_kernel_index(global_kernel_name);

  if(!kidx) {
    return make_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Could not obtain kernel index for kernel " +
                   global_kernel_name});
  }

  auto configuration_id = config.generate_id();
  int device = this->_dev.get_id();

  cuda_hardware_context *ctx = static_cast<cuda_hardware_context *>(
      this->_backend->get_hardware_manager()->get_device(device));

  std::string target_arch_name = ctx->get_device_arch();
  unsigned compute_capability = ctx->get_compute_capability();

  const hcf_kernel_info *kernel_info =
      rt::hcf_cache::get().get_kernel_info(hcf_object, kernel_name);
  if(!kernel_info) {
    return make_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Could not obtain hcf kernel info for kernel " +
            global_kernel_name});
  }

  auto code_object_selector = [&](const code_object *candidate) -> bool {
    if ((candidate->managing_backend() != backend_id::cuda) ||
        (candidate->source_compilation_flow() != compilation_flow::sscp) ||
        (candidate->state() != code_object_state::executable))
      return false;

    const cuda_sscp_executable_object *obj =
        static_cast<const cuda_sscp_executable_object *>(candidate);
    
    if(obj->configuration_id() != configuration_id)
      return false;

    return obj->get_device() == device;
  };

  auto code_object_constructor = [&]() -> code_object* {
    const common::hcf_container* hcf = rt::hcf_cache::get().get_hcf(hcf_object);
    
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        glue::jit::select_image(kernel_info, &kernel_names);

    // Construct PTX translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator = 
      compiler::createLLVMToPtxTranslator(kernel_names);

    // TODO Shouldn't we compile with the most recent ptx version supported
    // by clang/CUDA?
    translator->setBuildOption("ptx-version", compute_capability);
    translator->setBuildOption("ptx-target-device", compute_capability);

    // Lower kernels to PTX
    std::string ptx_image;
    auto err = glue::jit::compile(translator.get(),
        hcf, selected_image_name, config, ptx_image);
    
    if(!err.is_success()) {
      register_error(err);
      return nullptr;
    }

    cuda_sscp_executable_object *exec_obj = new cuda_sscp_executable_object{
        ptx_image, target_arch_name, hcf_object, kernel_names, device, config};
    result r = exec_obj->get_build_result();

    HIPSYCL_DEBUG_INFO
        << "cuda_queue: Successfully compiled SSCP kernels to module " << exec_obj->get_module()
        << std::endl;

    if(!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    return exec_obj;
  };

  const code_object *obj = kernel_cache::get().get_or_construct_code_object(
      *kidx, kernel_name, backend_id::cuda, hcf_object,
      code_object_selector, code_object_constructor);

  if(!obj) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Code object construction failed"});
  }

  CUmodule cumodule = static_cast<const cuda_executable_object*>(obj)->get_module();
  assert(cumodule);

  glue::jit::cxx_argument_mapper arg_mapper{*kernel_info, args, arg_sizes,
                                            num_args};
  if(!arg_mapper.mapping_available()) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "cuda_queue: Could not map C++ arguments to kernel arguments"});
  }
  return launch_kernel_from_module(cumodule, kernel_name, num_groups,
                                   group_size, local_mem_size, _stream,
                                   arg_mapper.get_mapped_args());

#else
  return make_error(
      __hipsycl_here(),
      error_info{
          "cuda_queue: SSCP kernel launch was requested, but hipSYCL was "
          "not built with CUDA SSCP support."});
#endif
}

device_id cuda_queue::get_device() const {
  return _dev;
}

void *cuda_queue::get_native_type() const {
  return static_cast<void*>(get_stream());
}

cuda_multipass_code_object_invoker::cuda_multipass_code_object_invoker(
    cuda_queue *q)
    : _queue{q} {}

result cuda_queue::query_status(inorder_queue_status &status) {
  auto err = cudaStreamQuery(_stream);
  if(err == cudaSuccess) {
    status = inorder_queue_status{true};
  } else if(err == cudaErrorNotReady) {
    status = inorder_queue_status{false};
  } else {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: Could not query stream status",
                                 error_code{"CU", static_cast<int>(err)}});
  }

  return make_success();
}

result cuda_multipass_code_object_invoker::submit_kernel(
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
      num_args);
}

result cuda_sscp_code_object_invoker::submit_kernel(
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

