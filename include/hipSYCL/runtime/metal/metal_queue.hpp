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

#include "metal_allocator.hpp"
#include "metal_event.hpp"

#include <mach/mach_time.h>

#include <vector>

namespace MTL {

class Device;
class Buffer;
class Resource;
class CommandBuffer;
class CommandQueue;
class ComputeCommandEncoder;
class SharedEvent;
class SharedEventListener;

} // namespace MTL

namespace hipsycl {
namespace rt {

// Returns nanoseconds in the same time domain as MTLCommandBuffer GPUStartTime/GPUEndTime
inline uint64_t metal_now_ns() {
  mach_timebase_info_data_t info;
  mach_timebase_info(&info);
  return static_cast<uint64_t>((static_cast<__uint128_t>(mach_absolute_time()) * info.numer) / info.denom);
}

class metal_submission_timestamp : public instrumentations::submission_timestamp {
public:
  metal_submission_timestamp(uint64_t now)
    : _time(profiler_clock::time_point{profiler_clock::duration{now}})
  {}

  metal_submission_timestamp()
    : _time{profiler_clock::time_point{profiler_clock::duration{metal_now_ns()}}}
  {}

  profiler_clock::time_point get_time_point() const override {
    return _time;
  }

  void wait() const override
  {}

private:
  profiler_clock::time_point _time;
};

template<class BaseInstrumentation>
class metal_gpu_timestamp : public BaseInstrumentation {
public:
  metal_gpu_timestamp() = default;
  metal_gpu_timestamp(uint64_t now) {
    set_time_ns(now);
  }

  void set_time(double seconds) {
    set_time_ns(static_cast<uint64_t>(seconds * 1e9));
  }

  void set_time_ns(uint64_t ns) {
    _time.store(ns, std::memory_order_release);
    _signal.signal();
  }

  profiler_clock::time_point get_time_point() const override {
    return profiler_clock::time_point{profiler_clock::duration{_time.load(std::memory_order_acquire)}};
  }

  void wait() const override {
    _signal.wait();
  }

private:
  std::atomic<uint64_t> _time{0};
  mutable signal_channel _signal;
};

using metal_execution_start_timestamp = metal_gpu_timestamp<instrumentations::execution_start_timestamp>;
using metal_execution_finish_timestamp = metal_gpu_timestamp<instrumentations::execution_finish_timestamp>;

struct metal_profiling_setup {
  std::shared_ptr<metal_execution_start_timestamp> start_time;
  std::shared_ptr<metal_execution_finish_timestamp> finish_time;
};

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
  metal_inorder_queue(MTL::Device* device, metal_allocator* allocator, const device_id& id, hardware_context* hw_ctx);
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

  // Makes all USM allocations known to the allocator resident for the given
  // encoder. Residency established this way is scoped to the encoder, so this
  // must be called for every encoder; what is cached across encoders is only
  // the *set* of allocations, which is rebuilt (taking the allocator lock)
  // exclusively when the allocator generation changes.
  // Public because it is invoked by the free kernel launch helper.
  void make_allocations_resident(MTL::ComputeCommandEncoder* encoder);

private:
  // Common setup for freshly created command buffers: encodes the wait on the
  // previous queue work and attaches pending profiling handlers.
  MTL::CommandBuffer* prepare_command_buffer(MTL::CommandBuffer* cmd_buf);

  // Returns the currently open (uncommitted) command buffer of this queue,
  // creating one if none exists. Operations append their encoders to the open
  // buffer; it is committed lazily by flush().
  MTL::CommandBuffer* get_open_command_buffer();

  // Batching bookkeeping to be invoked after an operation has been fully
  // encoded into the open buffer: enforces a bound on ops per command buffer
  // and closes the window for per-op GPU profiling timestamps.
  void finish_batched_op();

  // Creates a fresh command buffer that is NOT the open buffer, for operations
  // that must be committed immediately (e.g. device->host copies with CPU-side
  // completion logic). Serializes against all prior work on this queue.
  MTL::CommandBuffer* new_dedicated_command_buffer();

  // Commits the open command buffer, if any. Synchronization with subsequent
  // work is expressed exclusively through explicit events and host-visible
  // copies, matching the semantics of the non-batched submission path.
  result flush();

  void profiling_setup(operation& op, const dag_node_ptr& node);

  MTL::Device* _device = nullptr;
  MTL::CommandQueue* _command_queue = nullptr;
  MTL::SharedEvent* _shared_event = nullptr;
  MTL::SharedEventListener* _event_listener = nullptr;
  // Atomic because query_status() and wait() may be called from arbitrary
  // threads concurrently with submit_*() / insert_event().
  std::atomic<uint64_t> _event_counter{0};

  // Only touched by submit_*() and insert_event(), which are serialized
  // by external mutex, so no atomics needed.
  uint64_t _pending_cpu_event{0};
  uint64_t _pending_gpu_event{0};

  // Command buffer batching: operations append encoders to _open_buffer which
  // is committed lazily (on wait/event/flush boundaries or when
  // max_ops_per_command_buffer is reached). Encoders within a command buffer
  // execute in encoding order, so batching preserves in-order semantics.
  // Retained manually; must not be a NS::SharedPtr because this header must
  // not include Foundation headers (see metal_hardware_manager.cpp, which
  // defines the metal-cpp private implementation symbols).
  MTL::CommandBuffer* _open_buffer = nullptr;
  bool _open_buffer_has_trailing_signal{false};
  int _open_op_count{0};
  bool _flush_after_current_op{false};

  static constexpr int max_ops_per_command_buffer = 32;

  // Cached snapshot of the allocations that must be made resident, together
  // with the allocator generation it was taken at. Rebuilt only when the
  // allocator generation changes; independent of command buffer boundaries.
  std::vector<const MTL::Resource*> _resident_resources;
  uint64_t _residency_generation{~static_cast<uint64_t>(0)};

  metal_allocator* _allocator = nullptr;
  device_id _device_id;

  metal_sscp_code_object_invoker _sscp_code_object_invoker;

  std::shared_ptr<kernel_cache> _kernel_cache;

  glue::jit::reflection_map _reflection_map;

  glue::jit::cxx_argument_mapper _arg_mapper;

  kernel_configuration _config;

  common::spin_lock _sscp_submission_spin_lock;

  std::optional<metal_profiling_setup> _profiling_setup;
};

} // namespace rt
} // namespace hipsycl

#endif // HIPSYCL_METAL_EXECUTOR_HPP
