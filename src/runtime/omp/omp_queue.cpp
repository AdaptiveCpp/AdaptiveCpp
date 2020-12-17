/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay and contributors
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

namespace {

bool is_contigous(id<3> offset, range<3> r, range<3> allocation_shape) {
  if (r.size() == 0)
    return true;
  
  int dim = 3;
  if (r.get(0) == 1)
    dim = 2;
  if (r.get(1) == 1)
    dim = 1;

  // 1D data transfers are always contiguous
  if (dim == 1)
    return true;

  // The slowest index does not need to be of 0 offset and
  // full size => start at
  // * 2 for dim == 2 (slowest index is 1)
  // * 1 for dim == 3 (slowest index is 0)
  for (int i = 4 - dim; i <= 2; ++i) {
    if (offset.get(i) != 0)
      return false;
    if (r.get(i) != allocation_shape.get(i))
      return false;
  }

  return true;
}

}

omp_queue::omp_queue(backend_id id)
: _backend_id(id) {}

omp_queue::~omp_queue() {
  _worker.halt();
}

std::unique_ptr<dag_node_event> omp_queue::insert_event() {
  HIPSYCL_DEBUG_INFO << "omp_queue: Inserting event into queue..." << std::endl;
  
  auto evt = std::make_unique<omp_node_event>();
  auto signal_channel = evt->get_signal_channel();

  _worker([signal_channel]{
    signal_channel->signal();
  });

  return evt;
}

std::unique_ptr<omp_timestamp_profiler> omp_queue::begin_profiling(const operation &op) {
  if (!op.is_instrumented() || !op.get_instrumentations().is_instrumented<rt::timestamp_profiler>())
    return nullptr;
  auto profiler = std::make_unique<omp_timestamp_profiler>();
  profiler->record_submit();
  return profiler;
}

void omp_queue::finish_profiling(operation &op, std::unique_ptr<omp_timestamp_profiler> profiler) {
  if (!profiler) return;
  op.get_instrumentations().provide<rt::timestamp_profiler>(std::move(profiler));
}

result omp_queue::submit_memcpy(memcpy_operation &op) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting memcpy operation..." << std::endl;

  if (op.source().get_device().is_host() && op.dest().get_device().is_host()) {

    void* base_src = op.source().get_base_ptr();
    void *base_dest = op.dest().get_base_ptr();
    
    assert(base_src);
    assert(base_dest);

    range<3> transferred_range = op.get_num_transferred_elements();
    range<3> src_allocation_shape = op.source().get_allocation_shape();
    range<3> dest_allocation_shape = op.dest().get_allocation_shape();
    id<3> src_offset = op.source().get_access_offset();
    id<3> dest_offset = op.dest().get_access_offset();
    std::size_t src_element_size = op.source().get_element_size();
    std::size_t dest_element_size = op.dest().get_element_size();

    std::size_t total_num_bytes = op.get_num_transferred_bytes();

    bool is_src_contiguous =
        is_contigous(src_offset, transferred_range, src_allocation_shape);
    bool is_dest_contiguous =
        is_contigous(dest_offset, transferred_range, dest_allocation_shape);

    auto profiler = begin_profiling(op);

    _worker([=, profiler=profiler.get()]() {
      if (profiler) profiler->record_start();

      auto linear_index = [](id<3> id, range<3> allocation_shape) {
        return id[2] + allocation_shape[2] * id[1] +
               allocation_shape[2] * allocation_shape[1] * id[0];
      };

      if (is_src_contiguous && is_dest_contiguous) {
        char *current_src = reinterpret_cast<char *>(base_src);
        char *current_dest = reinterpret_cast<char *>(base_dest);
        
        current_src +=
            linear_index(src_offset, src_allocation_shape) * src_element_size;
        current_dest +=
            linear_index(dest_offset, dest_allocation_shape) *
                dest_element_size;

        memcpy(current_dest, current_src, total_num_bytes);
      } else {
        id<3> current_src_offset = src_offset;
        id<3> current_dest_offset = dest_offset;
        std::size_t row_size = transferred_range[2] * src_element_size;

        for (std::size_t surface = 0; surface < transferred_range[0]; ++surface) {
          for (std::size_t row = 0; row < transferred_range[1]; ++row) {

            char *current_src = reinterpret_cast<char *>(base_src);
            char *current_dest = reinterpret_cast<char *>(base_dest);

            current_src +=
                linear_index(current_src_offset, src_allocation_shape) *
                src_element_size;

            current_dest +=
                linear_index(current_dest_offset, dest_allocation_shape) *
                dest_element_size;

            assert(current_src + row_size <=
                   reinterpret_cast<char *>(base_src) +
                       src_allocation_shape.size() * src_element_size);
            assert(current_dest + row_size <=
                   reinterpret_cast<char *>(base_dest) +
                       dest_allocation_shape.size() * dest_element_size);
            
            memcpy(current_dest, current_src, row_size);

            ++current_src_offset[1];
            ++current_dest_offset[1];
          }
          current_src_offset[1] = src_offset[1];
          current_dest_offset[1] = dest_offset[1];

          ++current_dest_offset[0];
          ++current_src_offset[0];
        }
      }

      if (profiler) profiler->record_finish();
    });

    finish_profiling(op, std::move(profiler));

  } else {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: OpenMP CPU backend cannot transfer data between "
                   "host and accelerators.",
                   error_type::feature_not_supported});
  }

  return make_success();
}

result omp_queue::submit_kernel(kernel_operation &op) {
  HIPSYCL_DEBUG_INFO << "omp_queue: Submitting kernel..." << std::endl;

  rt::backend_kernel_launcher *launcher = 
      op.get_launcher().find_launcher(_backend_id);

  if(!launcher) {
    return register_error(
        __hipsycl_here(),
        error_info{"omp_queue: Could not find required kernel launcher",
        error_type::runtime_error});
  }

  auto profiler = begin_profiling(op);

  _worker([=, profiler=profiler.get()]() {
    HIPSYCL_DEBUG_INFO << "omp_queue [async]: Invoking kernel!" << std::endl;
    if (profiler) profiler->record_start();
    launcher->invoke();
    if (profiler) profiler->record_finish();
  });

  finish_profiling(op, std::move(profiler));
  return make_success();
}

result omp_queue::submit_prefetch(prefetch_operation &op) {
  HIPSYCL_DEBUG_INFO
      << "omp_queue: Received prefetch submission request, ignoring"
      << std::endl;
  // Yeah, what are you going to do? Prefetching CPU memory on CPU? Go home!
  // (TODO: maybe we should handle the case that we have USM memory from another
  // backend here)
  if (op.is_instrumented() && op.get_instrumentations().is_instrumented<rt::timestamp_profiler>())
    op.get_instrumentations().provide<rt::timestamp_profiler>(omp_timestamp_profiler::make_no_op());
  return make_success();
}

result omp_queue::submit_memset(memset_operation & op) {
  void *ptr = op.get_pointer();
  std::size_t bytes = op.get_num_bytes();
  int pattern = op.get_pattern();
  
  if (!ptr) {
    return register_error(
        __hipsycl_here(),
        error_info{
            "omp_queue: submit_memset(): Invalid argument, pointer is null."});
  }

  auto profiler = begin_profiling(op);

  _worker([=, profiler=profiler.get()]() {
    if (profiler) profiler->record_start();
    memset(ptr, pattern, bytes);
    if (profiler) profiler->record_finish();
  });

  finish_profiling(op, std::move(profiler));
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
