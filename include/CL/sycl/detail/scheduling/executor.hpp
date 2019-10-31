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

#ifndef HIPSYCL_EXECUTOR_HPP
#define HIPSYCL_EXECUTOR_HPP

#include <unordered_map>
#include <memory>
#include <vector>
#include "device_id.hpp"
#include "hints.hpp"

#include "../../backend/backend.hpp"

namespace cl {
namespace sycl {
namespace detail {

class backend_executor
{
public:
  virtual bool is_inorder_queue() const = 0;
  virtual bool is_outoforder_queue() const = 0;
  virtual bool is_taskgraph() const = 0;

  virtual execution_hints get_default_execution_hints() const = 0;

  virtual ~backend_executor(){}
};

class inorder_queue : public backend_executor
{
public:
  bool is_inorder_queue() const final override;
  bool is_outoforder_queue() const final override;
  bool is_taskgraph() const final override;
};



using execution_context_ptr = std::unique_ptr<backend_executor>;

class executor_manager
{
public:
  void initialize_device(device_id d);

  std::size_t get_num_executors(device_id d)
  {
    // ToDo: Maybe initialize device when no queues are yet present?
    return _executors[d].size();
  }

  backend_executor* get_queue(device_id d, std::size_t index) const
  {
    
  }
private:
  std::unordered_map<device_id, std::vector<execution_context_ptr>> _executors;
};

}
}
}

#endif
