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
#ifndef HIPSYCL_INORDER_QUEUE_EVENT_HPP
#define HIPSYCL_INORDER_QUEUE_EVENT_HPP

#include "event.hpp"

namespace hipsycl {
namespace rt {

class inorder_queue;

template<class FineGrainedBackendEventT>
class inorder_queue_event : public dag_node_event {
public:
  // Get fine-grained backend event that should be inserted into
  // the queue as early as possible after the operation that it describes.
  //
  // This should only be invoked when absolutely necessary, since it can
  // force event implementations that rely on interacting with the backend only
  // lazily to actually insert events into a backend inorder queue.
  //
  // This function does not have to be thread-safe and should therefore only be
  // invoked inside the runtime thread.
  virtual FineGrainedBackendEventT request_backend_event() = 0;
  virtual ~inorder_queue_event() {}
};


}
}

#endif
