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
#ifndef HIPSYCL_EXECUTOR_HPP
#define HIPSYCL_EXECUTOR_HPP

#include "dag_node.hpp"
#include "device_id.hpp"
#include "operations.hpp"
#include "hints.hpp"


namespace hipsycl {
namespace rt {


struct backend_execution_lane_range
{
  std::size_t begin;
  std::size_t num_lanes;
};

class backend_executor
{
public:

  virtual bool is_inorder_queue() const = 0;
  virtual bool is_outoforder_queue() const = 0;
  virtual bool is_taskgraph() const = 0;

  virtual void
  submit_directly(const dag_node_ptr& node, operation *op,
                  const node_list_t &reqs) = 0;

  virtual bool can_execute_on_device(const device_id& dev) const = 0;
  virtual bool is_submitted_by_me(const dag_node_ptr& node) const = 0;

  virtual ~backend_executor(){}
};



}
}

#endif
