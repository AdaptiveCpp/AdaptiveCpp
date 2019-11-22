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


#include "hints.hpp"


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


  /// Inserts an event into the stream
  virtual std::unique_ptr<dag_node_event> insert_event() = 0;

  virtual void submit_memcpy(const memcpy_operation&) = 0;
  virtual void submit_kernel(const kernel_operation&) = 0;
  virtual void submit_prefetch(const prefetch_operation&) = 0;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue may be from the same or a different backend.
  virtual void submit_queue_wait_for(dag_node_event*) = 0;
};






};

}
}
}

#endif
