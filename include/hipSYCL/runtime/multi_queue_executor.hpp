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

#ifndef HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP
#define HIPSYCL_MULTI_QUEUE_EXECUTOR_HPP

#include <cmath>
#include <cassert>
#include <functional>

#include "backend.hpp"
#include "device_id.hpp"
#include "executor.hpp"
#include "inorder_queue.hpp"
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
  get_memcpy_execution_lane_range(device_id dev) const override;

  // The range of lanes to use for the given device
  backend_execution_lane_range
  get_kernel_execution_lane_range(device_id dev) const override;

  virtual void
  submit_directly(dag_node_ptr node, operation *op,
                  const std::vector<dag_node_ptr> &reqs) override;

  template <class F> void for_each_queue(rt::device_id dev, F handler) const {
    assert(dev.get_id() < _device_data.size());
    for (std::size_t i = 0; i < _device_data[dev.get_id()].queues.size(); ++i)
      handler(_device_data[dev.get_id()].queues[i].get());
  }
private:
  using final_nodes_map = std::unordered_map<inorder_queue*, dag_node_ptr>;

  struct per_device_data
  {
    backend_execution_lane_range memcpy_lanes;
    backend_execution_lane_range kernel_lanes;
    std::vector<std::unique_ptr<inorder_queue>> queues;

    moving_statistics submission_statistics;
  };

  std::vector<per_device_data> _device_data;
  std::size_t _num_submitted_operations;
};

}
}


#endif
