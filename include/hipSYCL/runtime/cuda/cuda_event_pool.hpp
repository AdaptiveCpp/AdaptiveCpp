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
#ifndef HIPSYCL_CUDA_EVENT_POOL_HPP
#define HIPSYCL_CUDA_EVENT_POOL_HPP

#include "../event_pool.hpp"
#include "cuda_event.hpp"


namespace hipsycl {
namespace rt {

class cuda_event_factory {
public:
  using event_type = CUevent_st*;
  cuda_event_factory(int device_id);

  result create(event_type&);
  result destroy(event_type);
private:
  int _device_id;
};

class cuda_event_pool : public event_pool<cuda_event_factory> {
public:
  cuda_event_pool(int device_id);
};

}
}

#endif