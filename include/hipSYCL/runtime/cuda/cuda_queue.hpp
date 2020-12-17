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

#ifndef HIPSYCL_CUDA_QUEUE_HPP
#define HIPSYCL_CUDA_QUEUE_HPP

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "cuda_instrumentation.hpp"

// Forward declare CUstream_st instead of including cuda_runtime_api.h.
// It's not possible to include both HIP and CUDA headers since they
// define conflicting symbols. Therefore we should not include
// cuda_runtime_api.h in runtime header files.
// Note: CUstream_st* == cudaStream_t.
struct CUstream_st;
// Note: CUevent_st* == cudaEvent_t
struct CUevent_st;

namespace hipsycl {
namespace rt {

class cuda_queue : public inorder_queue
{
public:
  cuda_queue(device_id dev);

  CUstream_st* get_stream() const;

  virtual ~cuda_queue();

  /// Inserts an event into the stream
  virtual std::unique_ptr<dag_node_event> insert_event() override;

  virtual result submit_memcpy(memcpy_operation&) override;
  virtual result submit_kernel(kernel_operation&) override;
  virtual result submit_prefetch(prefetch_operation &) override;
  virtual result submit_memset(memset_operation&) override;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) override;
  virtual result submit_external_wait_for(dag_node_ptr node) override;

  device_id get_device() const { return _dev; }
private:
  void activate_device() const;

  std::unique_ptr<cuda_timestamp_profiler> begin_profiling(const operation &op) const;
  void finish_profiling(operation &op, std::unique_ptr<cuda_timestamp_profiler> profiler) const;

  device_id _dev;
  CUstream_st* _stream;
  cuda_timestamp_profiler::baseline _profiler_baseline;
};

}
}

#endif
