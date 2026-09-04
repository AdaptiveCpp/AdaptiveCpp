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
#include "hipSYCL/runtime/vk/vk_queue.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/queue_completion_event.hpp"
#include "hipSYCL/runtime/vk/vk_allocator.hpp"
#include "hipSYCL/runtime/vk/vk_event.hpp"
#include "hipSYCL/runtime/vk/vk_hardware_manager.hpp"

#ifdef HIPSYCL_WITH_SSCP_COMPILER
#include "hipSYCL/compiler/llvm-to-backend/clspv/LLVMToCLSPV.hpp"
#include "hipSYCL/compiler/llvm-to-backend/clspv/LLVMToCLSPVFactory.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/adaptivity_engine.hpp"
#endif

namespace hipsycl {
namespace rt {

vk_queue::vk_queue(vk_hardware_manager *hw_manager, std::size_t device_index)
    : _hw_manager{hw_manager}, _device_index{device_index}, _cmd_bufs(nullptr),
      _timeline_value(0), _semaphore(nullptr), _sscp_invoker{this},
      _kernel_cache{kernel_cache::get()} {
  _dev_ctx =
      static_cast<vk_hardware_context *>(hw_manager->get_device(device_index));
  auto &device = _dev_ctx->get_device();

  _dev_has_khr_calibrated_timestamps = _dev_ctx->are_extensions_enabled(
      vk_device_extensions::khr_calibrated_timestamps);

  _reflection_map = glue::jit::construct_default_reflection_map(_dev_ctx);

  vk::CommandPoolCreateInfo pool_info{
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      _dev_ctx->get_queue_index()};
  _cmd_pool = std::make_unique<vk::raii::CommandPool>(device, pool_info);

  vk::CommandBufferAllocateInfo cmd_buf_alloc_info(
      *_cmd_pool.get(), vk::CommandBufferLevel::ePrimary, _cmd_buf_alloc_size);
  _cmd_bufs = vk::raii::CommandBuffers(device, cmd_buf_alloc_info);

  // Add all newly allocated command-buffers to the list of ready-to-use
  // command-buffers
  for (vk::CommandBuffer cb : _cmd_bufs) {
    _available_cmd_bufs.push_back(cb);
  }

  // Create a timeline semaphore
  vk::StructureChain<vk::SemaphoreCreateInfo, vk::SemaphoreTypeCreateInfo>
      semaphore_info({}, {vk::SemaphoreType::eTimeline, _timeline_value});
  _semaphore =
      device.createSemaphore(semaphore_info.get<vk::SemaphoreCreateInfo>());
}

std::shared_ptr<dag_node_event> vk_queue::create_queue_completion_event() {
  return std::make_shared<queue_completion_event<vk::Semaphore, vk_node_event>>(
      this);
}

std::shared_ptr<dag_node_event> vk_queue::insert_event() {
  // Create event representing when the last command submitted to the queue has
  // completed
  return std::make_shared<vk_node_event>(this, _timeline_value);
}

// Queue keeps a list of device pointers of allocations, but allocator keeps
// the actual allocations, then queue uses dev pointer to free alloc
std::pair<vk_alloc_info *, bool>
vk_queue::find_or_create_allocation(vk::DeviceAddress ptr, unsigned size) {
  vk_allocator *allocator = _dev_ctx->get_allocator();
  vk_alloc_info *alloc_info = allocator->find_alloc_info(ptr);
  if (alloc_info) {
    return std::make_pair(alloc_info, false);
  }

  auto dev_ptr = allocator->raw_allocate(0 /* ignore alignment*/, size);
  alloc_info =
      allocator->find_alloc_info(reinterpret_cast<vk::DeviceAddress>(dev_ptr));
  return std::make_pair(alloc_info, true);
}

void vk_queue::profile_if_enabled(operation &op, const dag_node_ptr &node) {
  if (!node) {
    return;
  }

  auto &hints = node->get_execution_hints();
  if (hints.has_hint<
          rt::hints::request_instrumentation_submission_timestamp>()) {
    op.get_instrumentations()
        .add_instrumentation<instrumentations::submission_timestamp>(
            std::make_shared<vk_sync_timestamp>(
                _dev_ctx->get_device(), _dev_has_khr_calibrated_timestamps));
  }

  vk_async_profiling async_prof;
  if (hints.has_hint<rt::hints::request_instrumentation_start_timestamp>()) {
    async_prof.start_time = std::make_shared<vk_execution_start_timestamp>(
        _dev_ctx->get_device(), *_semaphore,
        _dev_has_khr_calibrated_timestamps);
    op.get_instrumentations()
        .add_instrumentation<instrumentations::execution_start_timestamp>(
            async_prof.start_time);
  }

  if (hints.has_hint<rt::hints::request_instrumentation_finish_timestamp>()) {
    async_prof.finish_time = std::make_shared<vk_execution_finish_timestamp>(
        _dev_ctx->get_device(), *_semaphore,
        _dev_has_khr_calibrated_timestamps);
    op.get_instrumentations()
        .add_instrumentation<instrumentations::execution_finish_timestamp>(
            async_prof.finish_time);
  }

  // Reset this for every new command submission, the underlying
  // instrumentation object from previous command submissions is kept alive
  // through the shared pointer attached to `op`.
  if (async_prof.start_time || async_prof.finish_time) {
    _profiling = std::move(async_prof);
  } else {
    _profiling.reset();
  }
}

result vk_queue::submit_memcpy(memcpy_operation &op, const dag_node_ptr &node) {
  id<3> src_offset = op.source().get_access_offset();
  id<3> dest_offset = op.dest().get_access_offset();
  const range<3> transfer_range = op.get_num_transferred_elements();
  const vk::DeviceAddress src_ptr =
      reinterpret_cast<vk::DeviceAddress>(op.source().get_access_ptr());
  const vk::DeviceAddress dst_ptr =
      reinterpret_cast<vk::DeviceAddress>(op.dest().get_access_ptr());
  const vk::DeviceSize size = op.get_num_transferred_bytes();

  int dimension = 0;
  {
    if (transfer_range[0] > 1)
      dimension = 3;
    else if (transfer_range[1] > 1)
      dimension = 2;
    else
      dimension = 1;
  }

  // If we transfer the entire buffer, treat it as 1D memcpy for performance.
  if (op.get_num_transferred_elements() == op.source().get_allocation_shape() &&
      op.get_num_transferred_elements() == op.dest().get_allocation_shape() &&
      op.source().get_access_offset() == id<3>{} &&
      op.dest().get_access_offset() == id<3>{})
    dimension = 1;

  assert(dimension >= 1 && dimension <= 3);

  profile_if_enabled(op, node);

  auto dst_alloc = find_or_create_allocation(dst_ptr, size);
  auto src_alloc = find_or_create_allocation(src_ptr, size);
  std::pair<vk_alloc_info *, vk_alloc_info *> temp_allocs{nullptr, nullptr};

  // We need to copy host data into the new src buffer.
  if (src_alloc.second) {
    const uint64_t wait_value = _timeline_value;
    const uint64_t signal_value = ++_timeline_value;

    vk_alloc_info *src_alloc_info = src_alloc.first;
    _host_worker(
        [=]() mutable {
          vk::Semaphore semaphore = *_semaphore;

          HIPSYCL_DEBUG_INFO
              << "vk_queue: temp allocation source copy async thread WAIT "
              << "semaphore " << semaphore << " wait value " << wait_value
              << std::endl;
          vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &wait_value);
          vk::Result wait_ret_code;
          do {
            wait_ret_code =
                _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
          } while (vk::Result::eTimeout == wait_ret_code);

          if (wait_ret_code != vk::Result::eSuccess) {
            std::string err_msg(
                "Semaphore wait failed with unexpected return code ");
            err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
            print_error(__acpp_here(), error_info{err_msg});
          }

          if (_profiling && _profiling->start_time) {
            // Since we're dong work before command buffer starts executing,
            // use an earlier host timestamp
            _profiling->start_time->take_host_timestamp();
          }

          void *vptr = src_alloc_info->_dev_mem.mapMemory(0, size);
          std::memcpy(vptr, reinterpret_cast<void *>(src_ptr), size);
          src_alloc_info->_dev_mem.unmapMemory();

          vk::SemaphoreSignalInfo signal_info(semaphore, signal_value);
          _dev_ctx->get_device().signalSemaphore(signal_info);

          HIPSYCL_DEBUG_INFO
              << "vk_queue: temp allocation source copy async thread SIGNAL "
              << "semaphore " << semaphore << " signal value " << signal_value
              << std::endl;
        });

    temp_allocs.first = src_alloc_info;
  }

  if (dst_alloc.second) {
    temp_allocs.second = dst_alloc.first;
  }

  // Track temporary buffer created in map
  if (temp_allocs.first || temp_allocs.second) {
    const uint64_t signal_value = _timeline_value + 1;
    _temp_allocs.insert(signal_value, temp_allocs);
  }

  // Append a copy-buffer command for every strided copy.
  vk::CommandBuffer cmd_buf =
      begin_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  std::vector<vk::BufferCopy> copy_regions;
  if (dimension == 1) {
    size_t x_src_offset = src_offset[0];
    if (x_src_offset == 0 && !src_alloc.second) {
      x_src_offset = src_ptr - src_alloc.first->_base_ptr;
    }

    size_t x_dst_offset = dest_offset[0];
    if (x_dst_offset == 0 && !dst_alloc.second) {
      x_dst_offset = dst_ptr - dst_alloc.first->_base_ptr;
    }
    copy_regions.emplace_back(x_src_offset, x_dst_offset, size);
  } else {
    const std::size_t src_element_size = op.source().get_element_size();
    const std::size_t dest_element_size = op.dest().get_element_size();
    const range<3> src_allocation_shape = op.source().get_allocation_shape();
    const range<3> dest_allocation_shape = op.dest().get_allocation_shape();

    auto linear_index = [](id<3> id, range<3> allocation_shape) {
      return id[2] + allocation_shape[2] * id[1] +
             allocation_shape[2] * allocation_shape[1] * id[0];
    };

    id<3> current_src_offset = src_offset;
    id<3> current_dest_offset = dest_offset;
    const std::size_t row_size = transfer_range[2] * src_element_size;
    for (std::size_t surface = 0; surface < transfer_range[0]; ++surface) {
      for (std::size_t row = 0; row < transfer_range[1]; ++row) {
        size_t src_offset =
            linear_index(current_src_offset, src_allocation_shape) *
            src_element_size;
        size_t dest_offset =
            linear_index(current_dest_offset, dest_allocation_shape) *
            dest_element_size;

        assert(src_offset + row_size <=
               src_allocation_shape.size() * src_element_size);
        assert(dest_offset + row_size <=
               dest_allocation_shape.size() * dest_element_size);

        copy_regions.emplace_back(src_offset, dest_offset, row_size);

        ++current_src_offset[1];
        ++current_dest_offset[1];
      }
      current_src_offset[1] = src_offset[1];
      current_dest_offset[1] = dest_offset[1];

      ++current_dest_offset[0];
      ++current_src_offset[0];
    }
  }

  cmd_buf.copyBuffer(src_alloc.first->_buffer, dst_alloc.first->_buffer,
                     copy_regions);
  end_command_buffer(cmd_buf);
  submit_command_buffer(cmd_buf);

  // Cleanup to be wrapped in async call, this is effectively an extra command
  // that follows a memcpy if we had to create a temporary allocation to
  // do the memcopy. It is required to free the allocations and copy back
  // the data in a temporary destination buffer to user pointer.
  if (temp_allocs.first || temp_allocs.second) {
    const uint64_t wait_value = _timeline_value;
    const uint64_t signal_value = ++_timeline_value;

    if (_profiling && _profiling->finish_time) {
      // Since we need to do work after the command-buffer finishes executing
      // override semaphore value to wait on
      _profiling->finish_time->set_semaphore_wait_val(signal_value);
    }

    _host_worker(
        [=]() mutable {
          vk::Semaphore semaphore = *_semaphore;
          HIPSYCL_DEBUG_INFO
              << "vk_queue: temp allocation deallocate async thread WAIT "
              << "semaphore " << semaphore << " wait value " << wait_value
              << std::endl;
          vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &wait_value);

          vk::Result wait_ret_code;
          do {
            wait_ret_code =
                _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
          } while (vk::Result::eTimeout == wait_ret_code);

          if (wait_ret_code != vk::Result::eSuccess) {
            std::string err_msg(
                "Semaphore wait failed with unexpected return code ");
            err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
            print_error(__acpp_here(), error_info{err_msg});
          }

          auto temp_alloc_pair = _temp_allocs.get(wait_value);

          // pair is <source operand, dest operand>,
          vk_allocator *allocator = _dev_ctx->get_allocator();
          if (auto dst_alloc = temp_alloc_pair.second; dst_alloc != nullptr) {
            void *vptr = dst_alloc->_dev_mem.mapMemory(0, size);
            std::memcpy(reinterpret_cast<void *>(dst_ptr), vptr, size);
            dst_alloc->_dev_mem.unmapMemory();
            allocator->raw_free(reinterpret_cast<void *>(dst_alloc->_base_ptr));
          }

          if (auto src_alloc = temp_alloc_pair.first; src_alloc != nullptr) {
            allocator->raw_free(reinterpret_cast<void *>(src_alloc->_base_ptr));
          }
          _temp_allocs.erase(wait_value);

          if (_profiling && _profiling->finish_time) {
            // Since we're dong work after command buffer starts executing,
            // use a later host timestamp
            _profiling->finish_time->take_host_timestamp();
          }

          vk::SemaphoreSignalInfo signal_info(semaphore, signal_value);
          _dev_ctx->get_device().signalSemaphore(signal_info);

          HIPSYCL_DEBUG_INFO
              << "vk_queue: temp allocation deallocate async thread SIGNAL "
              << "semaphore " << semaphore << " signal value " << signal_value
              << std::endl;
        });
  }

  return make_success();
}

vk::CommandBuffer
vk_queue::begin_command_buffer(vk::CommandBufferUsageFlagBits flags) {
  auto cmd_buf = get_command_buffer();
  cmd_buf.begin(vk::CommandBufferBeginInfo(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

  if (_profiling && _profiling->start_time) {
    // When profiling, start the command buffer with commands to reset and
    // write a query pool timestamp. This gives the device side timestamp
    // for when the command begins execution.
    auto query_pool = _profiling->start_time->get_query_pool();

    cmd_buf.resetQueryPool(query_pool, 0, 1);
    cmd_buf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                           query_pool, 0);
  }
  return cmd_buf;
}

void vk_queue::end_command_buffer(vk::CommandBuffer &cmd_buf) {
  if (_profiling && _profiling->finish_time) {
    // When profiling, end the command buffer with commands to reset and
    // write a query pool timestamp. This gives the device side timestamp
    // for when the command finishes execution.
    auto query_pool = _profiling->finish_time->get_query_pool();
    cmd_buf.resetQueryPool(query_pool, 0, 1);
    cmd_buf.writeTimestamp(vk::PipelineStageFlagBits::eComputeShader,
                           query_pool, 0);
  }
  cmd_buf.end();
}

vk::CommandBuffer vk_queue::get_command_buffer() {
  /*  Overview of command-buffer management:
   *
   *  - queue maintains RAII _cmd_bufs member backing the lifetime
   *   of allocated command-buffers
   *  - Two non-owning containers for command buffers are also kept which
   *   reference the objects in the owning RAII list. A list of
   *   'available' command-buffers to use and a map of 'busy'
   *   command buffers with their timeline semaphore submission
   *   signal counter as the value.
   *
   * Algorithm for getting a command-buffer:
   * 1) Try take a new command-buffer off the available list.
   * 2) If free-list is empty then refresh busy-list by
   *    checking current semaphore counter against values
   *    in busy list. If we have matched for passed the
   *    current counter value, return the command-buffer
   *    to available list.
   * 3) If free list non-empty take a command-buffer.
   * 4) Otherwise wait on lowest signal value to become free.
   *
   * We could allocate more command-buffers from the pool, but
   * extending our RAII vector could invalidate the references
   * in the other lists, so leave this as future work to implement.
   */
  if (!_available_cmd_bufs.empty()) {
    HIPSYCL_DEBUG_INFO << "vk_queue: Found free command-buffer" << std::endl;
    vk::CommandBuffer cmd_buf = _available_cmd_bufs.back();
    _available_cmd_bufs.pop_back();
    return cmd_buf;
  }

  HIPSYCL_DEBUG_INFO << "vk_queue: No free command-buffers, refreshing list"
                     << std::endl;

  std::vector<uint64_t> finished;
  for (auto &pair : _executing_cmd_bufs) {
    auto &signal_val = pair.first;
    auto &cmd_buf = pair.second;
    if (_semaphore.getCounterValue() >= signal_val) {
      finished.push_back(signal_val);
      cmd_buf.reset({});
      _available_cmd_bufs.push_back(cmd_buf);
    }
  }

  for (auto &signal_val : finished) {
    _executing_cmd_bufs.erase(signal_val);
  }

  // Check if any command-buffers now available
  if (!_available_cmd_bufs.empty()) {
    HIPSYCL_DEBUG_INFO << "vk_queue: " << _available_cmd_bufs.size()
                       << " free command-buffers found in refreshed list"
                       << std::endl;
    vk::CommandBuffer cmd_buf = _available_cmd_bufs.back();
    _available_cmd_bufs.pop_back();
    return cmd_buf;
  }

  // Using an ordered map with signal value as the key, so first item
  // will have soonest value to wait on.
  auto cmd = _executing_cmd_bufs.begin();
  auto wait_val = cmd->first;
  auto cmd_buf = cmd->second;

  HIPSYCL_DEBUG_INFO << "vk_queue: Still no free command-buffers, waiting on "
                     << wait_val
                     << " counter to be signaled by timeline semaphore"
                     << std::endl;

  vk::Semaphore semaphore = *_semaphore;
  vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &wait_val);
  vk::Result wait_ret_code;
  do {
    wait_ret_code =
        _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
  } while (vk::Result::eTimeout == wait_ret_code);

  if (wait_ret_code != vk::Result::eSuccess) {
    std::string err_msg("Semaphore wait failed with unexpected return code ");
    err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
    print_error(__acpp_here(), error_info{err_msg});
  }

  _executing_cmd_bufs.erase(wait_val);
  cmd_buf.reset({});
  return cmd_buf;
}

void vk_queue::submit_command_buffer(vk::CommandBuffer &cmd_buf) {
  // Set new timeline semaphore values
  const uint64_t wait_value = _timeline_value;
  const uint64_t signal_value = ++_timeline_value;

  if (_profiling) {
    // Only let a user read the timestamps after the signal semaphore has fired
    if (_profiling->start_time) {
      _profiling->start_time->set_semaphore_wait_val(signal_value);
    }
    if (_profiling->finish_time) {
      _profiling->finish_time->set_semaphore_wait_val(signal_value);
    }
  }

  HIPSYCL_DEBUG_INFO << "vk_queue: submit command-buffer with "
                     << "semaphore " << *_semaphore << " wait value "
                     << wait_value << " & signal value " << signal_value
                     << std::endl;

  std::vector<vk::Semaphore> semaphores{_semaphore};
  std::vector<uint64_t> wait_values{wait_value};
  for (auto &evt : _wait_deps) {
    auto vk_evt = static_cast<vk_node_event *>(evt.get());
    const auto external_semaphore = vk_evt->get_event();
    semaphores.push_back(external_semaphore);
    const uint64_t external_signal_val = vk_evt->get_signal_val();
    wait_values.push_back(external_signal_val);

    HIPSYCL_DEBUG_INFO << "vk_queue: command-buffer submit extra wait "
                       << "on semaphore " << external_semaphore
                       << " with signal val " << external_signal_val
                       << std::endl;
  }

  const uint32_t num_wait_semaphores =
      static_cast<uint32_t>(wait_values.size());
  vk::TimelineSemaphoreSubmitInfo timeline_info{
      num_wait_semaphores, wait_values.data(), 1u, &signal_value};
  std::vector<vk::PipelineStageFlags> wait_stages(
      num_wait_semaphores, vk::PipelineStageFlagBits::eComputeShader);
  vk::SubmitInfo submit_info(num_wait_semaphores, semaphores.data(),
                             wait_stages.data(), 1, &cmd_buf, 1,
                             semaphores.data());
  vk::StructureChain<vk::SubmitInfo, vk::TimelineSemaphoreSubmitInfo>
      submit_info_chain(submit_info, timeline_info);

  _dev_ctx->get_queue().submit(submit_info_chain.get<vk::SubmitInfo>(),
                               nullptr);
  // Track the command-buffer and signal val as executing
  _executing_cmd_bufs.insert({signal_value, cmd_buf});

  // Wipe event deps from other queues as we have now respected these
  _wait_deps.clear();
}

result vk_queue::submit_kernel(kernel_operation &op, const dag_node_ptr &node) {
  rt::backend_kernel_launch_capabilities cap;
  cap.provide_sscp_invoker(&_sscp_invoker);
  profile_if_enabled(op, node);
  return op.get_launcher().invoke(backend_id::vk, this, cap, node.get());
}

result vk_queue::submit_prefetch(prefetch_operation &op,
                                 const dag_node_ptr &node) {
  profile_if_enabled(op, node);
  vk::CommandBuffer cmd_buf =
      begin_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  // Empty command buffer, ignore perf hint as no-op
  end_command_buffer(cmd_buf);

  submit_command_buffer(cmd_buf);

  return make_success();
}

result vk_queue::submit_memset(memset_operation &op, const dag_node_ptr &node) {
  // In order to deal with sizes not divisible by 4 we implement memset on host
  // rather than using a command-buffer command.
  int pattern = op.get_pattern();
  size_t size = op.get_num_bytes();
  auto ptr = reinterpret_cast<vk::DeviceAddress>(op.get_pointer());
  auto ptr_alloc = find_or_create_allocation(ptr, size);
  // assert we're only dealing with user created allocations
  assert(!ptr_alloc.second);

  profile_if_enabled(op, node);

  // Need to snapshot these values now, as _timeline_value may changed
  // by the time the async function is invoked.
  const uint64_t wait_value = _timeline_value;
  const uint64_t signal_value = ++_timeline_value;

  if (_profiling) {
    // Only let a user read the timestamps after the signal semaphore has fired
    if (_profiling->start_time) {
      _profiling->start_time->set_semaphore_wait_val(signal_value);
    }
    if (_profiling->finish_time) {
      _profiling->finish_time->set_semaphore_wait_val(signal_value);
    }
  }

  _host_worker([=]() mutable {
    vk::Semaphore semaphore = *_semaphore;

    HIPSYCL_DEBUG_INFO << "vk_queue: memset async thread WAIT "
                       << "semaphore " << semaphore << " wait value "
                       << wait_value << std::endl;
    vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &wait_value);
    vk::Result wait_ret_code;
    do {
      wait_ret_code =
          _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
    } while (vk::Result::eTimeout == wait_ret_code);

    if (wait_ret_code != vk::Result::eSuccess) {
      std::string err_msg("Semaphore wait failed with unexpected return code ");
      err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
      print_error(__acpp_here(), error_info{err_msg});
    }

    if (_profiling && _profiling->start_time) {
      _profiling->start_time->take_host_timestamp();
    }

    char *vptr = (char *)ptr_alloc.first->_dev_mem.mapMemory(0, size);
    auto offset = ptr - ptr_alloc.first->_base_ptr;
    std::memset(vptr + offset, pattern, size);
    ptr_alloc.first->_dev_mem.unmapMemory();

    if (_profiling && _profiling->finish_time) {
      _profiling->finish_time->take_host_timestamp();
    }

    vk::SemaphoreSignalInfo signal_info(semaphore, signal_value);
    _dev_ctx->get_device().signalSemaphore(signal_info);

    HIPSYCL_DEBUG_INFO << "vk_queue: memset async thread SIGNAL "
                       << "semaphore " << semaphore << " signal value "
                       << signal_value << std::endl;
  });

  return make_success();
}

result vk_queue::wait() {
  vk::Semaphore semaphore = *_semaphore;

  HIPSYCL_DEBUG_INFO << "vk_queue: wait on semaphore " << semaphore
                     << " wait value " << _timeline_value << std::endl;

  vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &_timeline_value);
  vk::Result wait_ret_code;
  do {
    wait_ret_code =
        _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
  } while (vk::Result::eTimeout == wait_ret_code);

  if (wait_ret_code != vk::Result::eSuccess) {
    std::string err_msg("Semaphore wait failed with unexpected return code ");
    err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
    print_error(__acpp_here(), error_info{err_msg});
  }

  return make_success();
}

result vk_queue::submit_queue_wait_for(const dag_node_ptr &node) {
  auto evt = node->get_event();
  _wait_deps.push_back(evt);
  return make_success();
}

result vk_queue::submit_external_wait_for(const dag_node_ptr &node) {
  const uint64_t wait_value = _timeline_value;
  const uint64_t signal_value = ++_timeline_value;

  HIPSYCL_DEBUG_INFO << "vk_queue: external wait with semaphore wait val "
                     << wait_value << " and signal val " << signal_value
                     << std::endl;

  _host_worker([=]() mutable {
    vk::Semaphore semaphore = *_semaphore;

    HIPSYCL_DEBUG_INFO << "vk_queue: external wait async thread WAIT "
                       << "semaphore " << semaphore << " wait value "
                       << wait_value << std::endl;

    // Wait on in-order queue deps
    vk::SemaphoreWaitInfo wait_info({}, 1, &semaphore, &wait_value);
    vk::Result wait_ret_code;
    do {
      wait_ret_code =
          _dev_ctx->get_device().waitSemaphores(wait_info, UINT64_MAX);
    } while (vk::Result::eTimeout == wait_ret_code);

    if (wait_ret_code != vk::Result::eSuccess) {
      std::string err_msg("Semaphore wait failed with unexpected return code ");
      err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
      print_error(__acpp_here(), error_info{err_msg});
    }

    // Wait on external deps to complete
    node->wait();

    // All deps satisfied, signal this wait as completed
    vk::SemaphoreSignalInfo signal_info(semaphore, signal_value);
    _dev_ctx->get_device().signalSemaphore(signal_info);

    HIPSYCL_DEBUG_INFO << "vk_queue: external wait async thread SIGNAL "
                       << "semaphore " << semaphore << " signal value "
                       << signal_value << std::endl;
  });

  return make_success();
}

result vk_queue::query_status(inorder_queue_status &status) {
  const uint64_t counter = _semaphore.getCounterValue();
  status = inorder_queue_status{counter >= _timeline_value};
  return make_success();
}

device_id vk_queue::get_device() const {
  return _hw_manager->get_device_id(_device_index);
}

void *vk_queue::get_native_type() const {
  assert(false && "not implemented");
  HIPSYCL_DEBUG_WARNING << "vk_queue::get_native_type() not implemented"
                        << std::endl;
  return nullptr;
}

result vk_queue::submit_sscp_kernel_from_code_object(
    hcf_object_id hcf_object, std::string_view kernel_name,
    const rt::hcf_kernel_info *kernel_info, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const kernel_configuration &initial_config) {
#ifndef HIPSYCL_WITH_SSCP_COMPILER
  return make_error(
      __acpp_here(),
      error_info{"vk_queue: SSCP kernel launch was requested, but hipSYCL was "
                 "not built with Vulkan SSCP support."});
#else
  common::spin_lock_guard lock{_sscp_submission_spin_lock};

  _arg_mapper.construct_mapping(*kernel_info, args, arg_sizes, num_args);

  if (!_arg_mapper.mapping_available()) {
    return make_error(
        __acpp_here(),
        error_info{
            "vk_queue: Could not map C++ arguments to kernel arguments"});
  }

  kernel_adaptivity_engine adaptivity_engine{
      hcf_object, kernel_name, kernel_info, _arg_mapper, num_groups,
      group_size, args,        arg_sizes,   num_args,    local_mem_size};

  _config = initial_config;
  _config.append_base_configuration(kernel_base_config_parameter::backend_id,
                                    backend_id::vk);
  _config.append_base_configuration(
      kernel_base_config_parameter::compilation_flow, compilation_flow::sscp);
  _config.append_base_configuration(kernel_base_config_parameter::hcf_object_id,
                                    hcf_object);

  for (const auto &flag : kernel_info->get_compilation_flags())
    _config.set_build_flag(flag);
  for (const auto &opt : kernel_info->get_compilation_options())
    _config.set_build_option(opt.first, opt.second);

  _config.set_build_option(
      kernel_build_option::spirv_dynamic_local_mem_allocation_size,
      local_mem_size);

  auto binary_configuration_id =
      adaptivity_engine.finalize_binary_configuration(_config);
  auto code_object_configuration_id = binary_configuration_id;

  kernel_configuration::extend_hash(
      code_object_configuration_id,
      kernel_base_config_parameter::runtime_device, _dev_ctx->get_device());

  auto jit_compiler = [&](std::string &compiled_image) -> bool {
    std::vector<std::string> kernel_names;
    std::string selected_image_name =
        adaptivity_engine.select_image_and_kernels(&kernel_names);

    // Construct SPIR-V translator to compile the specified kernels
    std::unique_ptr<compiler::LLVMToBackendTranslator> translator =
        std::move(compiler::createLLVMToCLSPVTranslator(kernel_names));

    auto raw_translator = translator.get();
    raw_translator->setBuildOption(
        "-max-pushconstant-size",
        std::to_string(_dev_ctx->get_max_push_constant_size()));
    raw_translator->setBuildOption(
        "-max-ubo-size",
        std::to_string(_dev_ctx->get_max_uniform_buffer_range()));
    raw_translator->setBuildOption("-device-name", _dev_ctx->get_device_name());

    // Lower kernels to SPIR-V
    bool enable_dead_arg_elimination = kernel_names.size() == 1;
    rt::result err = glue::jit::compile_and_store_stats(
        raw_translator, hcf_object, selected_image_name, _config,
        binary_configuration_id, _reflection_map, compiled_image,
        enable_dead_arg_elimination);

    if (!err.is_success()) {
      register_error(err);
      return false;
    }
    return true;
  };

  auto code_object_constructor =
      [&](const std::string &compiled_image) -> code_object * {
    vk_executable_object *exec_obj =
        new vk_executable_object{_dev_ctx, hcf_object, compiled_image, _config};
    result r = exec_obj->get_build_result();

    if (!r.is_success()) {
      register_error(r);
      delete exec_obj;
      return nullptr;
    }

    bool has_dead_arg_elimination =
        exec_obj->supported_backend_kernel_names().size() == 1;
    glue::jit::load_jit_output_metadata(*exec_obj, has_dead_arg_elimination,
                                        binary_configuration_id);

    return exec_obj;
  };

  const code_object *obj = _kernel_cache->get_or_construct_jit_code_object(
      code_object_configuration_id, binary_configuration_id, jit_compiler,
      code_object_constructor);

  if (!obj) {
    return make_error(__acpp_here(),
                      error_info{"vk_queue: Code object construction failed"});
  }

  if (obj->get_jit_output_metadata()
          .kernel_retained_arguments_indices.has_value()) {
    _arg_mapper.apply_dead_argument_elimination_mask(
        obj->get_jit_output_metadata()
            .kernel_retained_arguments_indices.value());
  }

  vk_kernel_object *kernel;
  result res = static_cast<const vk_executable_object *>(obj)->get_kernel(
      kernel_name, kernel);

  if (!res.is_success())
    return res;

  auto pipeline = kernel->create_pipeline(group_size);
  vk_kernel_uniform_descriptors &kernel_descriptors =
      kernel->create_kernel_descriptors();

  vk::CommandBuffer cmd_buf =
      begin_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  // command-buffer must be in the recording state before we set push constants
  pipeline->set_args(cmd_buf, kernel_descriptors, _arg_mapper);
  pipeline->bind(cmd_buf, kernel_descriptors);

  HIPSYCL_DEBUG_INFO << "vk_queue: Attempting to submit SSCP kernel"
                     << std::endl;
  cmd_buf.dispatch(num_groups[0], num_groups[1], num_groups[2]);
  end_command_buffer(cmd_buf);

  submit_command_buffer(cmd_buf);

  kernel_descriptors.set_completion_val(_semaphore, _timeline_value);
  on_kernel_launch_complete(kernel_name, obj);

  return make_success();
#endif
}
} // namespace rt
} // namespace hipsycl
