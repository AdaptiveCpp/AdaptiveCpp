/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay
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

#include "hipSYCL/runtime/hip/hip_code_object.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hip/hip_device_manager.hpp"
#include "hipSYCL/runtime/hip/hip_target.hpp"
#include <vector>

namespace hipsycl {
namespace rt {



hip_executable_object::~hip_executable_object() {
  if(_module) {
    hip_device_manager::get().activate_device(_device);

    auto err = hipModuleUnload(_module);

    if(err != hipSuccess) {
      register_error(
          __hipsycl_here(),
          error_info{"hip_executable_object: could not unload module",
                     error_code{"HIP", static_cast<int>(err)}});
    }
  }
}

hip_executable_object::hip_executable_object(hcf_object_id origin,
                                             const std::string &target,
                                             const std::string &hip_fat_binary,
                                             int device)
    : _origin{origin}, _target{target}, _device{device}, _module{nullptr} {
  _build_result = build(hip_fat_binary);
}

result hip_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state hip_executable_object::state() const {
  return code_object_state::executable;
}

code_format hip_executable_object::format() const {
  return code_format::native_isa;
}

backend_id hip_executable_object::managing_backend() const {
  return backend_id::hip;
}

hcf_object_id hip_executable_object::hcf_source() const {
  return _origin;
}

std::string hip_executable_object::target_arch() const {
  return _target;
}

std::vector<std::string>
hip_executable_object::supported_backend_kernel_names() const {
  // TODO This is currently not easily implementable. Should we throw an
  // exception? Remove it from the general interface?
  return {};
}

bool hip_executable_object::contains(const std::string &backend_kernel_name) const {
  if(!_build_result.is_success())
    return false;
  
  // Currently just implemented by trying to query a function.
  hipFunction_t f;
  auto err = hipModuleGetFunction(&f, _module, backend_kernel_name.c_str());

  // TODO We should actually figure out which error code exactly HIP
  // returns in case a kernel is not present to distinguish this from other errors.
  return err == hipSuccess;
}

hipModule_t hip_executable_object::get_module() const {
  return _module;
}

int hip_executable_object::get_device() const {
  return _device;
}

result hip_executable_object::build(const std::string& hip_fat_binary) {
  // It's unclear if this is actually needed for HIP?
  hip_device_manager::get().activate_device(_device);

  auto err = hipModuleLoadData(&_module, hip_fat_binary.c_str());

  if(err == hipSuccess)
    return make_success();
  else {
    return make_error(
        __hipsycl_here(),
        error_info{"hip_executable_object: could not create module",
                   error_code{"HIP", static_cast<int>(err)}});
  }
}

}
}