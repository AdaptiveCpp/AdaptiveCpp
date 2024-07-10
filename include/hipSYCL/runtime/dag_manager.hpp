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
#ifndef HIPSYCL_DAG_MANAGER_HPP
#define HIPSYCL_DAG_MANAGER_HPP

#include <mutex>

#include "dag.hpp"
#include "dag_builder.hpp"
#include "dag_direct_scheduler.hpp"
#include "dag_unbound_scheduler.hpp"
#include "dag_submitted_ops.hpp"
#include "generic/async_worker.hpp"


namespace hipsycl {
namespace rt {

class runtime;

class dag_manager
{
  friend class dag_build_guard;
public:
  dag_manager(runtime* rt);
  ~dag_manager();

  // Submits operations asynchronously
  void flush_async();
  // Submits operations asynchronously and
  // wait until they have been submitted
  void flush_sync();
  // Wait for completion of all submitted operations
  void wait();
  void wait(std::size_t node_group_id);

  node_list_t get_group(std::size_t node_group_id);

  void register_submitted_ops(dag_node_ptr);
private:
  void trigger_flush_opportunity();

  dag_builder* builder() const;

  std::unique_ptr<dag_builder> _builder;
  worker_thread _worker;
  
  dag_direct_scheduler _direct_scheduler;
  dag_unbound_scheduler _unbound_scheduler;
  dag_submitted_ops _submitted_ops;

  // Should only be used for flush_async()
  std::mutex _flush_mutex;

  // TODO: This is not used anywhere
  [[maybe_unused]] runtime* _rt;
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
