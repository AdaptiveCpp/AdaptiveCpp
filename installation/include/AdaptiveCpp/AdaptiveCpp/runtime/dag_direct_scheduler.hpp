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
#ifndef HIPSYCL_DAG_DIRECT_SCHEDULER_HPP
#define HIPSYCL_DAG_DIRECT_SCHEDULER_HPP

#include "dag_node.hpp"
#include "operations.hpp"

#include <functional>

namespace hipsycl {
namespace rt {

class runtime;

class dag_direct_scheduler {
public:
  dag_direct_scheduler(runtime* rt);
  void submit(dag_node_ptr node);

private:
  runtime* _rt;
};

}
}

#endif
