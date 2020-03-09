/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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
#include "hipSYCL/runtime/hip/hip_event.hpp"
#include "hipSYCL/runtime/hip/hip_error.hpp"
#include "hipSYCL/runtime/util.hpp"

#include <memory>

namespace hipsycl {
namespace rt {

namespace {

void host_synchronization_callback(hipStream_t stream, hipError_t status,
                                   void *userData) {
  
  assert(userData);
  dag_node_ptr* node = static_cast<dag_node_ptr*>(userData);
  // Don't leak if there's an error detected in check_error()
  if(status != hipSuccess) {
    delete node;
  }

  hip_check_error(status);

  (*node)->wait();
  delete node;
}

}

hip_queue::hip_queue(device_id dev) : _dev{dev} {
  hip_check_error(hipSetDevice(_dev.get_id()));
  hip_check_error(hipStreamCreateWithFlags(&_stream, hipStreamNonBlocking));
}

hipStream_t hip_queue::get_stream() const { return _stream; }

hip_queue::~hip_queue() { hip_check_error(hipStreamDestroy(_stream)); }

/// Inserts an event into the stream
std::unique_ptr<dag_node_event> hip_queue::insert_event() {
  hipEvent_t evt;
  hip_check_error(hipEventCreate(&evt));
  hip_check_error(hipEventRecord(evt, this->get_stream()));

  return std::make_unique<hip_node_event>(_dev, evt);
}

void hip_queue::submit_memcpy(const memcpy_operation&) {

}
void hip_queue::submit_kernel(const kernel_operation& op) {
  //op.get_launcher()
}
void hip_queue::submit_prefetch(const prefetch_operation&) {
  assert(false && "Unimplemented");
}

/// Causes the queue to wait until an event on another queue has occured.
/// the other queue must be from the same backend
void hip_queue::submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) {
  assert(dynamic_is<hip_node_event>(evt.get()));

  hip_node_event* hip_evt = cast<hip_node_event>(evt.get());
  hip_check_error(hipStreamWaitEvent(_stream, hip_evt->get_event(), 0));
}

void hip_queue::submit_external_wait_for(dag_node_ptr node) {

  dag_node_ptr* user_data = new dag_node_ptr;
  assert(user_data);
  *user_data = node;

  hip_check_error(
      hipStreamAddCallback(_stream, host_synchronization_callback,
                           reinterpret_cast<void *>(user_data), 0));
}

}
}

