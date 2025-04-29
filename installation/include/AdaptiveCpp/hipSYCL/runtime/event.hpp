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
#ifndef HIPSYCL_DAG_NODE_EVENT_HPP
#define HIPSYCL_DAG_NODE_EVENT_HPP

#include "device_id.hpp"
#include "error.hpp"

namespace hipsycl {
namespace rt {

class dag_node_event
{
public:
  virtual bool is_complete() const = 0;
  virtual void wait() = 0;
  virtual ~dag_node_event() {}
}; 

}
}


#endif
