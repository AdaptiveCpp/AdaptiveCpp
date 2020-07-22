/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#include "hipSYCL/runtime/omp/omp_hardware_manager.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"

namespace hipsycl {
namespace rt {


bool omp_hardware_context::is_cpu() const {
  return true;
}

bool omp_hardware_context::is_gpu() const {
  return false;
}

std::size_t omp_hardware_context::get_max_kernel_concurrency() const {
  return 1;
}
  
// TODO We could actually copy have more memcpy concurrency
std::size_t omp_hardware_context::get_max_memcpy_concurrency() const {
  return 1;
}

std::string omp_hardware_context::get_device_name() const {
  return "hipSYCL OpenMP host device";
}

std::string omp_hardware_context::get_vendor_name() const {
  return "the hipSYCL project";
}

std::size_t omp_hardware_manager::get_num_devices() const {
  return 1;
}

hardware_context* omp_hardware_manager::get_device(std::size_t index) {
  if(index != 0) {
    register_error(__hipsycl_here(),
                   error_info{"omp_hardware_manager: Requested device " +
                                  std::to_string(index) + " does not exist.",
                              error_type::invalid_parameter_error});
    return nullptr;
  }

  return &_device;
}

device_id omp_hardware_manager::get_device_id(std::size_t index) const {
  return device_id{
      backend_descriptor{hardware_platform::cpu, api_platform::openmp_cpu},
      static_cast<int>(index)};
}


}
}
