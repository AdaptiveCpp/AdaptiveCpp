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


#ifndef HIPSYCL_DAG_SCHEDULER_HPP
#define HIPSYCL_DAG_SCHEDULER_HPP

#include <vector>

#include "CL/sycl/detail/scheduling/operations.hpp"
#include "device_id.hpp"
#include "executor.hpp"

namespace cl {
namespace sycl {
namespace detail {

class node_scheduling_annotation 
{
public:
  using synchronization_op_ptr =
      std::shared_ptr<backend_synchronization_operation>;

  device_id get_target_device() const { return _execution_device; }
  void set_target_device(device_id target) { _execution_device = target; }

  std::size_t get_execution_lane() const { return _assigned_execution_lane; }
  void assign_to_execution_lane(backend_executor *executor, std::size_t lane)
  {
    _assigned_executor = executor;
    _assigned_execution_lane = lane;
  }

  backend_executor* get_executor() const { return _assigned_executor; }

  void add_synchronization_op(
      std::unique_ptr<backend_synchronization_operation> op)
  {
    _synchronization_ops.push_back(std::move(op));
  }

  const std::vector<synchronization_op_ptr>& get_synchronization_ops() const
  { return _synchronization_ops; }
private:
  device_id _execution_device;
  std::size_t _assigned_execution_lane;
  backend_executor* _assigned_executor;

  std::vector<synchronization_op_ptr> _synchronization_ops;
};

class dag;

class dag_scheduler
{
public:
  dag_scheduler();

  void submit(dag* d);
private:
  std::vector<device_id> _available_devices;
};

}
}
}

#endif
