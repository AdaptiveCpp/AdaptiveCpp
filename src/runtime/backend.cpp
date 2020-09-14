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

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hw_model/hw_model.hpp"
#include <algorithm>

#ifdef HIPSYCL_RT_ENABLE_HIP_BACKEND
#include "hipSYCL/runtime/hip/hip_backend.hpp"
#endif

#ifdef HIPSYCL_RT_ENABLE_CUDA_BACKEND
#include "hipSYCL/runtime/cuda/cuda_backend.hpp"
#endif

#ifdef HIPSYCL_RT_ENABLE_OMP_BACKEND
#include "hipSYCL/runtime/omp/omp_backend.hpp"
#endif

namespace hipsycl {
namespace rt {

backend_manager::backend_manager()
: _hw_model(std::make_unique<hw_model>(this))
{
#ifdef HIPSYCL_RT_ENABLE_HIP_BACKEND
  HIPSYCL_DEBUG_INFO << "backend_manager: Registering HIP backend..." << std::endl;
  _backends.push_back(std::make_unique<hip_backend>());
#endif
  
#ifdef HIPSYCL_RT_ENABLE_CUDA_BACKEND
  HIPSYCL_DEBUG_INFO << "backend_manager: Registering CUDA backend..." << std::endl;
  _backends.push_back(std::make_unique<cuda_backend>());
#endif
  
#ifdef HIPSYCL_RT_ENABLE_OMP_BACKEND
  HIPSYCL_DEBUG_INFO << "backend_manager: Registering OpenMP backend..." << std::endl;
  _backends.push_back(std::make_unique<omp_backend>());
#endif
}

backend_manager::~backend_manager()
{
  
}

backend *backend_manager::get(backend_id id) const {
  auto it = std::find_if(_backends.begin(), _backends.end(),
                         [id](const std::unique_ptr<backend> &b) -> bool {
                           return b->get_backend_descriptor().id == id;
                         });
  
  if(it == _backends.end()){
    register_error(
        __hipsycl_here(),
        error_info{"backend_manager: Requested backend is not available.",
                   error_type::runtime_error});

    return nullptr;
  }
  return it->get();
}

hw_model &backend_manager::hardware_model()
{
  return *_hw_model;
}

const hw_model &backend_manager::hardware_model() const 
{
  return *_hw_model;
}

}
}
