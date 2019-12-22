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

#include "CL/sycl/detail/scheduling/multi_queue_executor.hpp"

namespace cl {
namespace sycl {
namespace detail {

multi_queue_executor::multi_queue_executor(
    const backend &b, queue_factory_function queue_factory) 
{
  std::size_t num_devices = b.get_hardware_manager()->get_num_devices();

  std::size_t num_assigned_lanes = 0;

  _device_data.resize(num_devices);

  for (std::size_t dev = 0; dev < num_devices; ++dev) {

    device_id dev_id = b.get_hardware_manager()->get_device_id(dev);
    hardware_context *hw_context = b.get_hardware_manager()->get_device(dev);


    for (std::size_t i = 0; i < hw_context->get_max_memcpy_concurrency();
         ++i, ++num_assigned_lanes) {

      _device_data[dev].memcpy_queues.push_back(queue_factory(dev_id));
    }
  }
}
}
}
}