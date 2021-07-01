/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay and contributors
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

#ifndef HIPSYCL_INORDER_QUEUE_HPP
#define HIPSYCL_INORDER_QUEUE_HPP

#include <memory>
#include <string>

#include "dag_node.hpp"
#include "hints.hpp"
#include "operations.hpp"
#include "error.hpp"
#include "module_invoker.hpp"

namespace hipsycl {
namespace rt {

class inorder_queue
{
public:

  /// Inserts an event into the stream
  virtual std::shared_ptr<dag_node_event> insert_event() = 0;

  virtual result submit_memcpy(memcpy_operation&, dag_node_ptr) = 0;
  virtual result submit_kernel(kernel_operation&, dag_node_ptr) = 0;
  virtual result submit_prefetch(prefetch_operation &, dag_node_ptr) = 0;
  virtual result submit_memset(memset_operation&, dag_node_ptr) = 0;
  
  /// Causes the queue to wait until an event on another queue has occured.
  /// the other queue must be from the same backend
  virtual result submit_queue_wait_for(std::shared_ptr<dag_node_event> evt) = 0;
  virtual result submit_external_wait_for(dag_node_ptr node) = 0;

  virtual device_id get_device() const = 0;
  /// Return native type if supported, nullptr otherwise
  virtual void* get_native_type() const = 0;

  /// Get a module invoker to launch kernels from module images,
  /// if the backend supports this. Returns nullptr if unsupported.
  virtual module_invoker* get_module_invoker() = 0;

  virtual ~inorder_queue(){}
};

}
}

#endif
