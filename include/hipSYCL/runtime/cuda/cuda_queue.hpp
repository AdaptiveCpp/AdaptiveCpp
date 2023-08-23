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
#include "cuda_code_object.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/cuda/cuda_event.hpp"


// Forward declare CUstream_st instead of including cuda_runtime_api.h.
// It's not possible to include both HIP and CUDA headers since they
// define conflicting symbols. Therefore we should not include
// cuda_runtime_api.h in runtime header files.
// Note: CUstream_st* == cudaStream_t.
struct CUstream_st;

namespace hipsycl {
namespace rt {

class cuda_queue;
class cuda_backend;

class cuda_multipass_code_object_invoker
    : public multipass_code_object_invoker {
public:
  cuda_multipass_code_object_invoker(cuda_queue* q);

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name_tag,
                               const std::string &kernel_body_name) override;
  virtual ~cuda_multipass_code_object_invoker(){}
private:
  cuda_queue* _queue;
};

class cuda_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  cuda_sscp_code_object_invoker(cuda_queue* q)
  : _queue{q} {}

  virtual ~cuda_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name,
                               const glue::kernel_configuration& config) override;
private:
  cuda_queue* _queue;
};

class cuda_queue : public inorder_queue
{
public:
  cuda_queue(cuda_backend* be, device_id dev, int priority = 0);

  CUstream_st* get_stream() const;

  virtual ~cuda_queue();

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() override;
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  virtual result submit_memcpy(memcpy_operation &, dag_node_ptr) override;
  virtual result submit_kernel(kernel_operation &, dag_node_ptr) override;
  virtual result submit_prefetch(prefetch_operation &, dag_node_ptr) override;
  virtual result submit_memset(memset_operation &, dag_node_ptr) override;

  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(dag_node_ptr evt) override;
  virtual result submit_external_wait_for(dag_node_ptr node) override;

  virtual result wait() override;

  virtual device_id get_device() const override;

  virtual void *get_native_type() const override;

  virtual result query_status(inorder_queue_status& status) override;

  result submit_multipass_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &backend_kernel_name, const rt::range<3> &grid_size,
      const rt::range<3> &block_size, unsigned dynamic_shared_mem,
      void **kernel_args, std::size_t num_args);

  result submit_sscp_kernel_from_code_object(
      const kernel_operation &op, hcf_object_id hcf_object,
      const std::string &kernel_name, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const glue::kernel_configuration &config);

  const host_timestamped_event& get_timing_reference() const {
    return _reference_event;
  }
private:
  void activate_device() const;

  device_id _dev;
  CUstream_st *_stream;
  cuda_multipass_code_object_invoker _multipass_code_object_invoker;
  cuda_sscp_code_object_invoker _sscp_code_object_invoker;
  host_timestamped_event _reference_event;
  cuda_backend* _backend;
};

}
}

#endif
