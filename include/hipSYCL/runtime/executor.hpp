/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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
