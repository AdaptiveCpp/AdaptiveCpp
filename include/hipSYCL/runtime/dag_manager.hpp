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

#ifndef HIPSYCL_DAG_MANAGER_HPP
#define HIPSYCL_DAG_MANAGER_HPP

#include "dag.hpp"
#include "dag_builder.hpp"
#include "dag_direct_scheduler.hpp"
#include "dag_unbound_scheduler.hpp"
#include "dag_submitted_ops.hpp"
#include "generic/async_worker.hpp"


namespace hipsycl {
namespace rt {

class dag_interpreter;

class dag_manager
{
  friend class dag_build_guard;
public:
  dag_manager();
  ~dag_manager();

  // Submits operations asynchronously
  void flush_async();
  // Submits operations asynchronously and
  // wait until they have been submitted
  void flush_sync();
  // Wait for completion of all submitted operations
  void wait();
  void wait(std::size_t node_group_id);

  std::vector<dag_node_ptr> get_group(std::size_t node_group_id);

  void register_submitted_ops(dag_node_ptr);
private:
  void trigger_flush_opportunity();

  dag_builder* builder() const;

  std::unique_ptr<dag_builder> _builder;
  worker_thread _worker;
  
  dag_direct_scheduler _direct_scheduler;
  dag_unbound_scheduler _unbound_scheduler;
  dag_submitted_ops _submitted_ops;
};

class dag_build_guard
{
public:
  dag_build_guard(dag_manager& mgr)
  : _mgr{&mgr} {}

  ~dag_build_guard();

  dag_builder* builder() const
  { return _mgr->builder(); }
private:
  dag_manager* _mgr;
};

}
}

#endif