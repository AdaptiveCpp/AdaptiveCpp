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

#include "hipSYCL/runtime/hip/hip_queue.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include "hipSYCL/runtime/util.hpp"

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


}


void hip_queue::activate_device() const {
  hip_device_manager::get().activate_device(_dev.get_id());
}

hip_queue::hip_queue(device_id dev) : _dev{dev}, _stream{nullptr} {
  this->activate_device();

  auto err = hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking);
  if (err != hipSuccess) {
    register_error(__hipsycl_here(),
                   error_info{"hip_queue: Couldn't construct backend stream",
                              error_code{"HIP", err}});
    return;
  }

  _profiler_baseline = hip_timestamp_profiler::baseline::record(_stream);
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
std::unique_ptr<dag_node_event> hip_queue::insert_event() {
  this->activate_device();

  auto evt = make_hip_event();
  if (!evt) {
    return nullptr;
  }

  auto err = hipEventRecord(evt.get(), this->get_stream());
  if (err != hipSuccess) {
    register_error(
        __hipsycl_here(),
        error_info{"hip_queue: Couldn't record event", error_code{"HIP", err}});
    return nullptr;
  }

  return std::make_unique<hip_node_event>(_dev, std::move(evt));
}

std::unique_ptr<hip_timestamp_profiler> hip_queue::begin_profiling(const operation &op) const {
  if (!op.is_instrumented() || !op.get_instrumentations().is_instrumented<rt::timestamp_profiler>())
    return nullptr;
  this->activate_device();
  auto profiler = std::make_unique<hip_timestamp_profiler>(&_profiler_baseline);
  profiler->record_before_operation(_stream);
  return profiler;
}

void hip_queue::finish_profiling(operation &op,
                                 std::unique_ptr<hip_timestamp_profiler> profiler) const {
  if (!profiler) return;
  profiler->record_after_operation(_stream);
  op.get_instrumentations().provide<rt::timestamp_profiler>(std::move(profiler));
}

result hip_queue::submit_memcpy(memcpy_operation & op) {

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

  auto profiler = begin_profiling(op);

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

  finish_profiling(op, std::move(profiler));

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: Couldn't submit memcpy",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

result hip_queue::submit_kernel(kernel_operation &op) {

  this->activate_device();
  rt::backend_kernel_launcher *l =
      op.get_launcher().find_launcher(backend_id::hip);
  
  if (!l)
    return make_error(__hipsycl_here(), error_info{"Could not obtain backend kernel launcher"});
  l->set_params(this);

  auto profiler = begin_profiling(op);
  l->invoke();
  finish_profiling(op, std::move(profiler));

  return make_success();
}

result hip_queue::submit_prefetch(prefetch_operation& op) {

#ifdef HIPSYCL_RT_HIP_SUPPORTS_UNIFIED_MEMORY
  auto profiler = begin_profiling(op);
  hipError_t err = hipSuccess;
  
  if (op.get_target().is_host()) {
    err = hipMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                              hipCpuDeviceId, get_stream());
  } else {
    err = hipMemPrefetchAsync(op.get_pointer(), op.get_num_bytes(),
                              _dev.get_id(), get_stream());
  }
  finish_profiling(op, std::move(profiler));

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

result hip_queue::submit_memset(memset_operation &op) {

  auto profiler = begin_profiling(op);
  hipError_t err = hipMemsetAsync(op.get_pointer(), op.get_pattern(),
                                  op.get_num_bytes(), get_stream());
  finish_profiling(op, std::move(profiler));

  if (err != hipSuccess) {
    return make_error(__hipsycl_here(),
                      error_info{"hip_queue: hipMemsetAsync() failed",
                                 error_code{"HIP", err}});
  }

  return make_success();
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
result hip_queue::submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) {
  assert(dynamic_is<hip_node_event>(evt.get()));

  hip_node_event* hip_evt = cast<hip_node_event>(evt.get());
  auto err = hipStreamWaitEvent(_stream, hip_evt->get_event(), 0);
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

}
}

