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
#include "../generic/host_timestamped_event.hpp"

#include "cuda_instrumentation.hpp"
#include "cuda_module.hpp"


// Forward declare CUstream_st instead of including cuda_runtime_api.h.
// It's not possible to include both HIP and CUDA headers since they
// define conflicting symbols. Therefore we should not include
// cuda_runtime_api.h in runtime header files.
// Note: CUstream_st* == cudaStream_t.
struct CUstream_st;

namespace hipsycl {
namespace rt {

class cuda_queue;

class cuda_module_invoker : public module_invoker {
public:
  cuda_module_invoker(cuda_queue *q);

  virtual result
  submit_kernel(module_id_t id, const std::string &module_variant,
                const std::string *module_image, const rt::range<3> &num_groups,
                const rt::range<3>& group_size, unsigned local_mem_size,
                void **args, std::size_t* arg_sizes, std::size_t num_args,
                const std::string &kernel_name_tag,
                const std::string &kernel_body_name) override;

private:
  cuda_queue* _queue;
};

class cuda_queue : public inorder_queue
{
public:
  cuda_queue(device_id dev);

  CUstream_st* get_stream() const;

  virtual ~cuda_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;

  virtual result submit_memcpy(memcpy_operation &, dag_node_ptr) override;
  virtual result submit_kernel(kernel_operation &, dag_node_ptr) override;
  virtual result submit_prefetch(prefetch_operation &, dag_node_ptr) override;
  virtual result submit_memset(memset_operation &, dag_node_ptr) override;

  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) override;
  virtual result submit_external_wait_for(dag_node_ptr node) override;

  virtual device_id get_device() const override;

  virtual void *get_native_type() const override;

  virtual module_invoker* get_module_invoker() override;
  
  result submit_kernel_from_module(cuda_module_manager &manager,
                                   const cuda_module &module,
                                   const std::string &kernel_name,
                                   const rt::range<3> &grid_size,
                                   const rt::range<3> &block_size,
                                   unsigned dynamic_shared_mem,
                                   void **kernel_args);

  const host_timestamped_event& get_timing_reference() const {
    return _reference_event;
  }
private:
  void activate_device() const;

  device_id _dev;
  CUstream_st *_stream;
  cuda_module_invoker _module_invoker;
  host_timestamped_event _reference_event;
};

}
}

#endif
