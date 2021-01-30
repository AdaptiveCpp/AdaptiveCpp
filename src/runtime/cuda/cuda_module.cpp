/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2021 Aksel Alpay
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

#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "hipSYCL/runtime/cuda/cuda_module.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

namespace {


void trim_left(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));
}

void trim_right_space_and_parenthesis(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
        return !std::isspace(ch) && (ch != '(');
    }).base(), s.end());
}


}

cuda_module::cuda_module(cuda_module_id_t module_id, const std::string &target,
                         const std::string &code_content)
    : _id{module_id}, _target{target}, _content{code_content} {

  std::istringstream code_stream(code_content);
  std::string line;

  while (std::getline(code_stream, line)) {

    std::string kernel_identifier = ".visible .entry";
    auto pos = line.find(kernel_identifier);

    if (pos != std::string::npos) {
      line = line.substr(pos+kernel_identifier.size());
      trim_left(line);
      trim_right_space_and_parenthesis(line);
      HIPSYCL_DEBUG_INFO << "Detected kernel in module: " << line << std::endl;
      _kernel_names.push_back(line);
    }
  }
}

const std::vector<std::string> &
cuda_module::get_kernel_names() const {
  return _kernel_names;
}

std::string cuda_module::get_content() const {
  return _content;
}

bool cuda_module::guess_kernel_name(const std::string &kernel_group_name,
                                    const std::string &kernel_component_name,
                                    std::string &guessed_name) const {
  
  bool found = false;
  for (auto candidate : get_kernel_names()) {
    if (candidate.find(kernel_group_name) != std::string::npos &&
        candidate.find(kernel_component_name) != std::string::npos) {

      if (found) {
        HIPSYCL_DEBUG_WARNING
            << "Encountered multiple candidates for kernels from group "
            << kernel_group_name << " with component: " << kernel_component_name
            << ": " << candidate << std::endl;
        HIPSYCL_DEBUG_WARNING << "Keeping initial guess: " << guessed_name
                              << std::endl;
      } else {
        guessed_name = candidate;
        found = true;
      }
    }
  }
  
  return found;
}

cuda_module_id_t cuda_module::get_id() const { return _id; }

const std::string &cuda_module::get_target() const { return _target; }

cuda_module_manager::cuda_module_manager(std::size_t num_devices)
    : _cuda_modules(num_devices, nullptr), _active_modules(num_devices, 0) {}

cuda_module_manager::~cuda_module_manager() {
  for (std::size_t i = 0; i < _cuda_modules.size(); ++i) {
    if (_cuda_modules[i]) {
      cuda_device_manager::get().activate_device(i);
      auto err = cuModuleUnload(_cuda_modules[i]);

      if (err != CUDA_SUCCESS) {
        register_error(__hipsycl_here(),
                      error_info{"cuda_module_manager: could not unload module",
                                 error_code{"CU", static_cast<int>(err)}});
      }
      _cuda_modules[i] = nullptr;
    }
  }
}

const cuda_module &
cuda_module_manager::obtain_module(cuda_module_id_t id,
                                   const std::string &target,
                                   const std::string &content) {
  for (const cuda_module &mod : _modules) {
    if (mod.get_id() == id && mod.get_target() == target) {
      return mod;
    }
  }

  _modules.push_back(cuda_module(id, target, content));
  return _modules.back();
}

result cuda_module_manager::load(rt::device_id dev, const cuda_module &module,
                                 CUmod_st *&out) {
  
  assert(dev.backend() == backend_id::cuda);

  int dev_id = dev.get_id();
  assert(dev_id < _cuda_modules.size());
  assert(dev_id < _active_modules.size());
  
  if (_cuda_modules[dev_id]) {
    if (_active_modules[dev_id] == module.get_id()) {
      // The right module is already loaded in this device context
      out = _cuda_modules[dev_id];
      return make_success();
    }
  }

  // Replace module with new one
  cuda_device_manager::get().activate_device(dev_id);

  if (_cuda_modules[dev_id]) {
    auto err = cuModuleUnload(_cuda_modules[dev_id]);
    _cuda_modules[dev_id] = nullptr;

    if (err != CUDA_SUCCESS) {
      return make_error(__hipsycl_here(),
                    error_info{"cuda_module_manager: could not unload module",
                                error_code{"CU", static_cast<int>(err)}});
    }
  }

  auto err = cuModuleLoadDataEx(
      &(_cuda_modules[dev_id]),
      static_cast<void *>(const_cast<char *>(module.get_content().c_str())),
      0, nullptr,
      nullptr);

  if (err != CUDA_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_module_manager: could not load module",
                                 error_code{"CU", static_cast<int>(err)}});
  }
  _active_modules[dev_id] = module.get_id();
  out = _cuda_modules[dev_id];

  return make_success();
}


} // namespace rt
} // namespace hipsycl
