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

#include "../instrumentation.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <vulkan/vulkan_raii.hpp>

namespace hipsycl {
namespace rt {

class vk_hardware_context;

// Helper function for vkGetCalibratedTimestampsKHR() with the timestamps
// needed for profiling
struct calibrated_timestamps {
  uint64_t clock_monotonic;
  uint64_t device;
};
calibrated_timestamps vk_get_calibrated_timestamps(const vk::raii::Device &,
                                                   bool);

// Helper function for calling vkGetQueryPoolResults on a single element
// query pool
uint64_t vk_get_query_pool_result(const vk::raii::QueryPool &);

// Helper function for creating a single element query pool
vk::raii::QueryPool vk_create_query_pool(const vk::raii::Device &);

// Synchronous timestamp used for submission metric
class vk_sync_timestamp : public instrumentations::submission_timestamp {
public:
  vk_sync_timestamp(const vk::raii::Device &dev, bool use_khr) {
    // Use VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR value which is already in the
    // unit of nanoseconds
    uint64_t now = vk_get_calibrated_timestamps(dev, use_khr).clock_monotonic;
    _time = profiler_clock::time_point{profiler_clock::duration{now}};
  }

  profiler_clock::time_point get_time_point() const override { return _time; }

  void wait() const override {}

private:
  profiler_clock::time_point _time;
};

// Asynchronous timestamp used for command start/finish metric
// This needs to support device (via vkCommandBuffer) or
// host (via host_worker thread)  async execution.
template <class Base> class vk_async_timestamp : public Base {
public:
  vk_async_timestamp(const vk::raii::Device &dev, vk::Semaphore sem,
                     bool use_khr)
      : _dev(dev), _use_khr(use_khr), _query_pool(nullptr), _host_timestamp(0),
        _sem(sem), _sem_wait_val(0) {
    // Snapshot the current host and device timestamps
    _ref_timestamps = vk_get_calibrated_timestamps(_dev, _use_khr);

    // Create a single element query pool owned by this instrumentation object
    _query_pool = vk_create_query_pool(_dev);
  }

  // Getter allows vkQueue to append command-buffer commands using this
  // query-pool
  vk::QueryPool get_query_pool() { return *_query_pool; }

  // Set the value of the timeline semaphore to wait on for command completion
  void set_semaphore_wait_val(uint64_t wait_val) { _sem_wait_val = wait_val; }

  // Called as part of asynchronous host execution.
  void take_host_timestamp() {
    _host_timestamp =
        vk_get_calibrated_timestamps(_dev, _use_khr).clock_monotonic;
  }

  profiler_clock::time_point get_time_point() const override {
    // If command was executed on host, directly convert to time point
    if (_host_timestamp != 0) {
      return profiler_clock::time_point{
          profiler_clock::duration{_host_timestamp}};
    }

    // If command was executed on device then use the difference between host
    // and device counters from the reference timestamp to inform how to
    // convert the device timestamp we captured into a time point.
    const uint64_t &host_ref = _ref_timestamps.clock_monotonic;
    const uint64_t &dev_ref = _ref_timestamps.device;

    // Get captured device timestamp from query pool in domain
    // VK_TIME_DOMAIN_DEVICE_KHR
    const int64_t device_time = vk_get_query_pool_result(_query_pool);

    // Convert device timestamp to host timepoint
    int64_t host_time;
    if (host_ref > dev_ref) {
      host_time = (host_ref - dev_ref) + device_time;
    } else {
      host_time = device_time - (dev_ref - host_ref);
    }
    return profiler_clock::time_point{profiler_clock::duration{host_time}};
  }

  void wait() const override {
    HIPSYCL_DEBUG_INFO << "vk_async_timestamp: semaphore " << _sem
                       << " wait on " << _sem_wait_val << std::endl;

    // Wait on completion of semaphore indicating async work has completed
    vk::SemaphoreWaitInfo wait_info({}, 1, &_sem, &_sem_wait_val);
    vk::Result wait_ret_code;
    do {
      wait_ret_code = _dev.waitSemaphores(wait_info, UINT64_MAX);
    } while (vk::Result::eTimeout == wait_ret_code);

    if (wait_ret_code != vk::Result::eSuccess) {
      std::string err_msg("Semaphore wait failed with unexpected return code ");
      err_msg += std::to_string(static_cast<VkResult>(wait_ret_code));
      print_error(__acpp_here(), error_info{err_msg});
    }
  }

private:
  // Members below used for device execution
  const vk::raii::Device &_dev;
  bool _use_khr;
  vk::raii::QueryPool _query_pool;
  calibrated_timestamps _ref_timestamps;

  // Members used to implement wait() method
  vk::Semaphore _sem;
  uint64_t _sem_wait_val;

  // Members below used for host execution
  uint64_t _host_timestamp;
};

using vk_execution_start_timestamp =
    vk_async_timestamp<instrumentations::execution_start_timestamp>;
using vk_execution_finish_timestamp =
    vk_async_timestamp<instrumentations::execution_finish_timestamp>;

struct vk_async_profiling {
  std::shared_ptr<vk_execution_start_timestamp> start_time;
  std::shared_ptr<vk_execution_finish_timestamp> finish_time;
};

} // namespace rt
} // namespace hipsycl
