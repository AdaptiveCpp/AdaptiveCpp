/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#ifndef HIPSYCL_METAL_EXECUTOR_HPP
#define HIPSYCL_METAL_EXECUTOR_HPP

#include "../executor.hpp"
#include "../inorder_queue.hpp"
#include "../kernel_cache.hpp"
#include "../queue_completion_event.hpp"

#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"

#include "metal_allocator.hpp"
#include "metal_event.hpp"

namespace MTL {

class Device;

} // namespace MTL

namespace hipsycl {
namespace rt {

class metal_inorder_queue;
class metal_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  metal_sscp_code_object_invoker(metal_inorder_queue* q)
  : _queue{q} {}

  virtual ~metal_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
    hcf_object_id hcf_object,
    const rt::range<3> &num_groups,
    const rt::range<3> &group_size,
    unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    std::string_view kernel_name,
    const rt::hcf_kernel_info* kernel_info,
    const kernel_configuration& config) override;
private:
  metal_inorder_queue* _queue;
};

class metal_inorder_queue : public inorder_queue
{
public:
  metal_inorder_queue(MTL::Device* device, metal_allocator* allocator, const device_id& id);
  virtual std::shared_ptr<dag_node_event> insert_event();
  virtual std::shared_ptr<dag_node_event> create_queue_completion_event();

  virtual result submit_memcpy(memcpy_operation&, const dag_node_ptr&);
  virtual result submit_kernel(kernel_operation&, const dag_node_ptr&);
  virtual result submit_prefetch(prefetch_operation &, const dag_node_ptr&);
  virtual result submit_memset(memset_operation&, const dag_node_ptr&);

  virtual result submit_queue_wait_for(const dag_node_ptr& evt);
  virtual result submit_external_wait_for(const dag_node_ptr& node);

  virtual result wait();

  virtual device_id get_device() const;
  virtual void* get_native_type() const;

  virtual result query_status(inorder_queue_status& status);

  virtual result submit_sscp_kernel_from_code_object(hcf_object_id hcf_object,
    std::string_view kernel_name, const rt::hcf_kernel_info *kernel_info,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const kernel_configuration &config);

  virtual ~metal_inorder_queue();

  worker_thread& get_worker();

private:
  MTL::Device* _device = nullptr;
  metal_allocator* _allocator = nullptr;
  device_id _device_id;

  metal_sscp_code_object_invoker _sscp_code_object_invoker;

  std::shared_ptr<kernel_cache> _kernel_cache;

  glue::jit::reflection_map _reflection_map;

  glue::jit::cxx_argument_mapper _arg_mapper;

  kernel_configuration _config;

  common::spin_lock _sscp_submission_spin_lock;

  worker_thread _worker;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_EXECUTOR_HPP
