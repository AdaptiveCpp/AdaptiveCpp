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
#include <vector>

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"

#include "hip_allocator.hpp"
#include "hip_queue.hpp"
#include "hip_hardware_manager.hpp"
#include "hip_event_pool.hpp"

#ifndef HIPSYCL_HIP_BACKEND_HPP
#define HIPSYCL_HIP_BACKEND_HPP

namespace hipsycl {
namespace rt {


class hip_backend : public backend
{
public:
  hip_backend();
  virtual api_platform get_api_platform() const override;
  virtual hardware_platform get_hardware_platform() const override;
  virtual backend_id get_unique_backend_id() const override;
  
  virtual backend_hardware_manager* get_hardware_manager() const override;
  virtual backend_executor* get_executor(device_id dev) const override;
  virtual backend_allocator *get_allocator(device_id dev) const override;

  virtual std::string get_name() const override;

  virtual ~hip_backend(){}

  virtual std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) override;

  hip_event_pool* get_event_pool(device_id dev) const;
private:
  mutable hip_hardware_manager _hw_manager;
  mutable lazily_constructed_executor<multi_queue_executor> _executor;
};

}
}


#endif
