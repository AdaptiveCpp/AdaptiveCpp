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
#ifndef HIPSYCL_OCL_EVENT_HPP
#define HIPSYCL_OCL_EVENT_HPP

#include "../inorder_queue_event.hpp"
#include <CL/opencl.hpp>

namespace hipsycl {
namespace rt {

class ocl_node_event : public inorder_queue_event<cl::Event>
{
public:
  using backend_event_type = cl::Event;
 
  ocl_node_event(device_id dev, cl::Event evt);
  ocl_node_event(device_id dev);

  ~ocl_node_event();

  virtual bool is_complete() const override;
  virtual void wait() override;

  backend_event_type get_event() const;
  device_id get_device() const;

  backend_event_type request_backend_event() override;
private:
  bool _is_empty = false;
  device_id _dev;
  backend_event_type _evt;
};

}
}


#endif