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
#ifndef HIPSYCL_DAG_UNBOUND_SCHEDULER_HPP
#define HIPSYCL_DAG_UNBOUND_SCHEDULER_HPP

#include "dag_node.hpp"
#include "dag_direct_scheduler.hpp"

namespace hipsycl {
namespace rt {

class runtime;

class dag_unbound_scheduler {
public:
  dag_unbound_scheduler(runtime* rt);

  void submit(dag_node_ptr node);
private:
  std::vector<device_id> _devices;
  rt::dag_direct_scheduler _direct_scheduler;
  runtime* _rt;
};

}
}

#endif
