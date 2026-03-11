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
#pragma once

#include "../inorder_queue.hpp"
#include "hipSYCL/common/spin_lock.hpp"
#include "hipSYCL/glue/llvm-sscp/jit-reflection/reflection_map.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/generic/async_worker.hpp"
#include "hipSYCL/runtime/vk/vk_code_object.hpp"
#include <mutex>
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_hardware_manager;
class vk_hardware_context;
class vk_alloc_info;

class vk_queue : public inorder_queue {
public:
  vk_queue(vk_hardware_manager *hw_manager, std::size_t device_index);

  std::shared_ptr<dag_node_event> insert_event() override;
  std::shared_ptr<dag_node_event> create_queue_completion_event() override;

  result submit_memcpy(memcpy_operation &, const dag_node_ptr &) override;
  result submit_kernel(kernel_operation &, const dag_node_ptr &) override;
  result submit_prefetch(prefetch_operation &, const dag_node_ptr &) override;
  result submit_memset(memset_operation &, const dag_node_ptr &) override;

  result submit_queue_wait_for(const dag_node_ptr &evt) override;
  result submit_external_wait_for(const dag_node_ptr &node) override;

  result wait() override;

  device_id get_device() const override;
  void *get_native_type() const override;
  vk_hardware_context *get_dev_ctx() { return _dev_ctx; }
  uint64_t get_semaphore_counter_value() const {
    return _semaphore.getCounterValue();
  }
  vk::Semaphore get_semaphore() { return _semaphore; }

  result query_status(inorder_queue_status &status) override;

  result submit_sscp_kernel_from_code_object(
      hcf_object_id hcf_object, std::string_view kernel_name,
      const rt::hcf_kernel_info *kernel_info, const rt::range<3> &num_groups,
      const rt::range<3> &group_size, unsigned local_mem_size, void **args,
      std::size_t *arg_sizes, std::size_t num_args,
      const kernel_configuration &config) override;

private:
  /*
   * Helper functions
   */
  vk::CommandBuffer get_command_buffer();
  void submit_command_buffer(vk::CommandBuffer &cmd_buf);

  std::pair<vk_alloc_info *, bool>
  find_or_create_allocation(vk::DeviceAddress ptr, unsigned size);

  /*
   * Members of tracking backend device
   */
  vk_hardware_manager *_hw_manager;
  const std::size_t _device_index;
  vk_hardware_context *_dev_ctx;

  /*
   * Command submission members
   */
  uint32_t _cmd_buf_alloc_size =
      24; // Batch size of command-buffers to allocate
  std::unique_ptr<vk::raii::CommandPool> _cmd_pool;
  vk::raii::CommandBuffers _cmd_bufs; // Owns the RAII lifetime
  std::vector<vk::CommandBuffer> _available_cmd_bufs;
  std::map<uint64_t, vk::CommandBuffer> _executing_cmd_bufs; // ordered

  // Maps command signal value to any temporary memory allocations
  // that need freed asynchronously when it completes
  struct protected_map {
  public:
    using ValueType = std::pair<vk_alloc_info *, vk_alloc_info *>;
    auto insert(uint64_t key, ValueType &val) {
      std::lock_guard<std::mutex> lock{_mutex};
      return _alloc_map.insert({key, val});
    }

    ValueType get(uint64_t wait_value) {
      std::lock_guard<std::mutex> lock{_mutex};
      return _alloc_map[wait_value];
    }

    void erase(uint64_t wait_value) { _alloc_map.erase(wait_value); }

  private:
    std::unordered_map<uint64_t, ValueType> _alloc_map;
    mutable std::mutex _mutex;
  } _temp_allocs;

  /*
   * Synchronization members
   */
  uint64_t _timeline_value = 0;
  vk::raii::Semaphore _semaphore;
  // Events that come from other queues from the same backend
  std::vector<std::shared_ptr<dag_node_event>> _wait_deps;
  worker_thread _host_worker;
  mutable std::mutex _mutex;

  /*
   * SSCP members
   */
  vk_sscp_code_object_invoker _sscp_invoker;
  std::shared_ptr<kernel_cache> _kernel_cache;
  common::spin_lock _sscp_submission_spin_lock;
  glue::jit::cxx_argument_mapper _arg_mapper;
  kernel_configuration _config;
  glue::jit::reflection_map _reflection_map;
};

} // namespace rt
} // namespace hipsycl
