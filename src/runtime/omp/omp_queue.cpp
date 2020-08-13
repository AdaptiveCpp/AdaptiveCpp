/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include "hipSYCL/runtime/omp/omp_queue.hpp"
#include "hipSYCL/runtime/event.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/runtime/omp/omp_event.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_launcher.hpp"
#include "hipSYCL/runtime/operations.hpp"
#include <memory>

namespace hipsycl {
namespace rt {

omp_queue::omp_queue(backend_id id)
: _backend_id(id) {}

omp_queue::~omp_queue() {
  _worker.halt();
}

std::unique_ptr<dag_node_event> omp_queue::insert_event() {
  HIPSYCL_DEBUG_INFO << "omp_queue: Inserting event into queue..." << std::endl;
  
  auto evt = std::make_unique<omp_node_event>();
  auto completion_flag = evt->get_completion_flag();

  _worker([completion_flag]{
    completion_flag->store(true);
  });

  return evt;
}

result omp_queue::submit_memcpy(const memcpy_operation &op) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting memcpy operation..." << std::endl;

  if (op.source().get_device().is_host() && op.dest().get_device().is_host()) {

    void* src = op.source().get_access_ptr();
    void *dest = op.dest().get_access_ptr();

    assert(src);
    assert(dest);

    std::size_t row_size = op.get_num_transferred_elements().get(2) *
                           op.source().get_element_size();
    std::size_t num_rows = op.get_num_transferred_elements().get(1);
    std::size_t num_surfaces = op.get_num_transferred_elements().get(0);

    std::size_t row_src_pitch = op.source().get_allocation_shape().get(2) *
                                op.source().get_element_size();
    std::size_t row_dest_pitch = op.dest().get_allocation_shape().get(2) *
                                op.dest().get_element_size();

    _worker([=]() {
      char *current_src = reinterpret_cast<char*>(src);
      char *current_dest = reinterpret_cast<char*>(dest);

      for (std::size_t surface = 0; surface < num_surfaces; ++surface) {
        for (std::size_t row = 0; row < num_rows; ++row) {
        
          memcpy(current_dest, current_src, row_size);
          current_src += row_src_pitch;
          current_dest += row_dest_pitch;
        }
      }

    });
  } else {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: OpenMP CPU backend cannot transfer data between "
                   "host and accelerators.",
                   error_type::feature_not_supported});
  }

  return make_success();
}

result omp_queue::submit_kernel(const kernel_operation &op) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting kernel..." << std::endl;

  rt::backend_kernel_launcher *launcher = 
      op.get_launcher().find_launcher(_backend_id);

  if(!launcher) {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: Could not find required kernel launcher",
        error_type::runtime_error});
  }

  _worker([=]() {
    HIPSYCL_DEBUG_INFO << "omp_queue [async]: Invoking kernel!" << std::endl;
    launcher->invoke();
  });

  return make_success();
}

result omp_queue::submit_prefetch(const prefetch_operation &) {
  HIPSYCL_DEBUG_INFO
      << "omp_queue: Received prefetch submission request, ignoring"
      << std::endl;
  // Yeah, what are you going to do? Prefetching CPU memory on CPU? Go home!
  // (TODO: maybe we should handle the case that we have USM memory from another
  // backend here)
  return make_success();
}
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
result omp_queue::submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting wait for other queue..." << std::endl;
  if(!evt) {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: event for synchronization is null.",
                   error_type::invalid_parameter_error});
  }

  _worker([=](){
    evt->wait();
  });

  return make_success();
}

result omp_queue::submit_external_wait_for(dag_node_ptr node) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting wait for external node..."
                     << std::endl;
  
  if(!node) {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: node for synchronization is null.",
                   error_type::invalid_parameter_error});
  }
  
  _worker([=](){
    node->wait();
  });

  return make_success();
}

worker_thread& omp_queue::get_worker(){
  return _worker;
}

}
}
