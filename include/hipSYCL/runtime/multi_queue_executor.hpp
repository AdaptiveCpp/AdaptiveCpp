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
#ifndef HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP
#define HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP

#include <cmath>
#include <cassert>
#include <functional>
#include <atomic>
#include <mutex>

#include "backend.hpp"
#include "device_id.hpp"
#include "executor.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "inorder_executor.hpp"
#include "generic/multi_event.hpp"

namespace hipsycl {
namespace rt {


class moving_statistics {
public:
  moving_statistics() = default;
  moving_statistics(std::size_t max_entries, std::size_t num_bins,
                    std::size_t time_to_forget)
      : _max_entries{max_entries},
        _num_bins{num_bins},
        _time_to_forget{time_to_forget} {}

  void insert(std::size_t bin) {
    submission new_submission {};
    
    new_submission.bin = bin;
    new_submission.timestamp = now();

    _last_submissions.push_back(new_submission);
    if(_last_submissions.size() > _max_entries) {

      std::size_t num_deletions = _last_submissions.size() - _max_entries;

      _last_submissions.erase(_last_submissions.begin(),
                              _last_submissions.begin() + num_deletions);
    }


  }

  std::size_t get_num_entries_in_bin(std::size_t bin) const {

    std::size_t count = 0;
    for(const auto& s : _last_submissions) {
      if(s.bin == bin) ++count;
    }
    return count;
  }

  template<class WeightFunc>
  std::vector<double> build_weighted_bins(WeightFunc w) const {
    std::vector<double> bins_out(_num_bins);
    for(const auto& s : _last_submissions) {
      bins_out[s.bin] += w(s.timestamp);
    }
    return bins_out;
  }

  std::vector<double>
  build_decaying_bins() const {
    std::size_t now = this->now();
    return build_weighted_bins([now,this](std::size_t timestamp) -> double {
      double age = static_cast<double>(now - timestamp);
      assert(age > 0.);
      return std::max(0.0, 1.0 - age / static_cast<double>(_time_to_forget));
    });
  }

private:
  std::size_t now() const {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch())
            .count();
  }

  struct submission {
    std::size_t timestamp;
    std::size_t bin;
  };

  std::size_t _max_entries;
  std::size_t _num_bins;
  std::size_t _time_to_forget;

  std::vector<submission> _last_submissions;
};

/// An executor that submits tasks by serializing them onto 
/// to multiple inorder queues (e.g. CUDA streams)
class multi_queue_executor : public backend_executor
{
public:
  using queue_factory_function =
      std::function<std::unique_ptr<inorder_queue>(device_id)>;

  multi_queue_executor(
      const backend& b,
      queue_factory_function queue_factory);

  virtual ~multi_queue_executor() {}

  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;

  // The range of lanes to use for the given device
  backend_execution_lane_range
  get_memcpy_execution_lane_range(device_id dev) const;

  // The range of lanes to use for the given device
  backend_execution_lane_range
  get_kernel_execution_lane_range(device_id dev) const;

  virtual void
  submit_directly(const dag_node_ptr& node, operation *op,
                  const node_list_t &reqs) override;

  template <class F> void for_each_queue(rt::device_id dev, F handler) const {
    assert(dev.get_id() < _device_data.size());
    for (std::size_t i = 0; i < _device_data[dev.get_id()].executors.size(); ++i)
      handler(_device_data[dev.get_id()].executors[i]->get_queue());
  }

  virtual bool can_execute_on_device(const device_id& dev) const override;
  virtual bool is_submitted_by_me(const dag_node_ptr& node) const override;

  bool find_assigned_lane_index(const dag_node_ptr& node, std::size_t& index_out) const;
private:
  

  struct per_device_data
  {
    backend_execution_lane_range memcpy_lanes;
    backend_execution_lane_range kernel_lanes;
    std::vector<std::unique_ptr<inorder_executor>> executors;

    moving_statistics submission_statistics;
  };

  std::vector<per_device_data> _device_data;
  std::vector<inorder_queue*> _managed_queues;
  backend_id _backend;
};

template<class Executor>
class lazily_constructed_executor {
public:
  template<class Factory>
  lazily_constructed_executor(Factory&& F)
  : _is_initialized{false}, _factory{std::forward<Factory>(F)} {}

  Executor* get() {
    if(_is_initialized.load(std::memory_order_acquire))
      return _ptr.get();
    else {
      std::lock_guard<std::mutex> lock{_mutex};
      if(!_is_initialized.load(std::memory_order_acquire)) {
        _ptr = _factory();
        _is_initialized.store(true, std::memory_order_release);
      }
      return _ptr.get();
    }
  }

private:
  std::atomic<bool> _is_initialized;
  std::mutex _mutex;
  std::function<std::unique_ptr<Executor>()> _factory;
  std::unique_ptr<Executor> _ptr = nullptr;
};

}
}


#endif
