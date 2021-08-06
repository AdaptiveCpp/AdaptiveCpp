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

#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/cuda/cuda_instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/cuda/cuda_queue.hpp"
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/serialization/serialization.hpp"
#include "hipSYCL/runtime/util.hpp"

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
    assert(_node);

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

}


void cuda_queue::activate_device() const {
  cuda_device_manager::get().activate_device(_dev.get_id());
}

cuda_queue::cuda_queue(device_id dev)
    : _dev{dev}, _module_invoker{this}, _stream{nullptr} {
  this->activate_device();

  auto err = cudaStreamCreateWithFlags(&_stream, cudaStreamNonBlocking);
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
  this->activate_device();

  cudaEvent_t evt;
  auto err = cudaEventCreate(&evt);
  if(err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Couldn't create event", error_code{"CUDA", err}});
    return nullptr;
  }

  err = cudaEventRecord(evt, this->get_stream());
  if (err != cudaSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"cuda_queue: Couldn't record event", error_code{"CUDA", err}});
    return nullptr;
  }

  return std::make_shared<cuda_node_event>(_dev, evt);
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

  
  cuda_instrumentation_guard instrumentation{this, op, node};
  l->invoke(node.get());

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
result cuda_queue::submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) {
  assert(dynamic_is<cuda_node_event>(evt.get()));

  cuda_node_event* cuda_evt = cast<cuda_node_event>(evt.get());
  auto err = cudaStreamWaitEvent(_stream, cuda_evt->get_event(), 0);
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

result cuda_queue::submit_kernel_from_module(cuda_module_manager &manager,
                                             const cuda_module &module,
                                             const std::string &kernel_name,
                                             const rt::range<3> &grid_size,
                                             const rt::range<3> &block_size,
                                             unsigned dynamic_shared_mem,
                                             void **kernel_args) {

  this->activate_device();

  CUmodule cumodule;
  result res = manager.load(_dev, module, cumodule);
  if (!res.is_success())
    return res;

  CUfunction f;
  CUresult err = cuModuleGetFunction(&f, cumodule, kernel_name.c_str());

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
                       dynamic_shared_mem, _stream, kernel_args, nullptr);

  if (err != CUDA_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_queue: could not submit kernel from module",
                                 error_code{"CU", static_cast<int>(err)}});
  }
  
  return make_success();
}

device_id cuda_queue::get_device() const {
  return _dev;
}

void *cuda_queue::get_native_type() const {
  return static_cast<void*>(get_stream());
}

module_invoker *cuda_queue::get_module_invoker() {
  return &_module_invoker;
}

cuda_module_invoker::cuda_module_invoker(cuda_queue *q) : _queue{q} {}

result cuda_module_invoker::submit_kernel(
    module_id_t id, const std::string &module_variant,
    const std::string *module_image, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const std::string &kernel_name_tag, const std::string &kernel_body_name) {

  assert(_queue);
  assert(module_image);

  cuda_backend *be =
      cast<cuda_backend>(&(application::get_backend(rt::backend_id::cuda)));

  HIPSYCL_DEBUG_INFO << "cuda_module_invoker: Obtaining module with id " << id
                     << " in variant '" << module_variant << "'" << std::endl;
  
  const cuda_module &code_module =
      be->get_module_manager().obtain_module(id, module_variant, *module_image);

  // This will hold the actual kernel name in the device image
  std::string kernel_name;
  // First check if there is a kernel in the module that matches
  // the expected explicitly named kernel name
  if (!code_module.guess_kernel_name("__hipsycl_kernel", kernel_name_tag,
                                     kernel_name)) {

    // We are dealing with an unnamed kernel, so check if we can find
    // a matching unnamed kernel
    if (!code_module.guess_kernel_name("__hipsycl_kernel", kernel_body_name,
                                       kernel_name)) {

      return rt::make_error(
          __hipsycl_here(),
          rt::error_info{"cuda_module_invoker: No matching CUDA kernel "
                         "found in module for kernel with name tag " +
                         kernel_name_tag + " and type " +
                         kernel_body_name});
    }
  }
  HIPSYCL_DEBUG_INFO
      << "cuda_module_invoker: Selected kernel from module for execution: "
      << kernel_name << std::endl;

  return _queue->submit_kernel_from_module(
      be->get_module_manager(), code_module, kernel_name, num_groups,
      group_size, local_mem_size, args);
}
}
}

