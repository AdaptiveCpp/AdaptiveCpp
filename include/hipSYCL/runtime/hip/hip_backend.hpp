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

#include <vector>

#include "../backend.hpp"
#include "../multi_queue_executor.hpp"

#include "hip_allocator.hpp"
#include "hip_queue.hpp"
#include "hip_hardware_manager.hpp"

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

private:
  mutable hip_hardware_manager _hw_manager;
  mutable multi_queue_executor _executor;
  mutable std::vector<hip_allocator> _allocators;
};

}
}


#endif
