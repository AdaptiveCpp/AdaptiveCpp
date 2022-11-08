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

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"
#include "hipSYCL/runtime/cuda/cuda_code_object.hpp"
#include "hipSYCL/runtime/cuda/cuda_device_manager.hpp"
#include "hipSYCL/runtime/device_id.hpp"
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
        return !std::isspace(ch) && (ch != '(') && (ch != ')');
    }).base(), s.end());
}

void unload_cuda_module(CUmod_st* module, int device) {
  if (module) {
    cuda_device_manager::get().activate_device(device);

    auto err = cuModuleUnload(module);

    if (err != CUDA_SUCCESS && 
        // It can happen that during shutdown of the CUDA
        // driver we cannot unload anymore.
        // TODO: Find a better solution
        err != CUDA_ERROR_DEINITIALIZED) {
      register_error(
          __hipsycl_here(),
          error_info{"cuda_executable_object: could not unload module",
                     error_code{"CU", static_cast<int>(err)}});
    }
  }
}

result build_cuda_module_from_ptx(CUmod_st *&module, int device,
                                  const std::string &source) {
  
  cuda_device_manager::get().activate_device(device);
  // This guarantees that the CUDA runtime API initializes the CUDA
  // context on that device. This is important for the subsequent driver
  // API calls which assume that CUDA context has been created.
  cudaFree(0);
  
  auto err = cuModuleLoadDataEx(
      &module, static_cast<void *>(const_cast<char *>(source.c_str())),
      0, nullptr, nullptr);

  if (err != CUDA_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"cuda_executable_object: could not load module",
                                error_code{"CU", static_cast<int>(err)}});
  }
  
  assert(module);

  return make_success();
}
}

cuda_source_object::cuda_source_object(hcf_object_id origin,
                                       const std::string &target,
                                       const std::string &source)
    : _origin{origin}, _target_arch{target}, _source{source} {

  std::istringstream code_stream(source);
  std::string line;

  while (std::getline(code_stream, line)) {

    const std::string kernel_identifier = ".visible .entry";
    auto pos = line.find(kernel_identifier);

    if (pos != std::string::npos) {
      line = line.substr(pos + kernel_identifier.size());
      trim_left(line);
      trim_right_space_and_parenthesis(line);
      HIPSYCL_DEBUG_INFO << "Detected kernel in code object: " << line
                         << std::endl;
      _kernel_names.push_back(line);
    }
  }
}

code_object_state cuda_source_object::state() const {
  return code_object_state::source;
}

code_format cuda_source_object::format() const { return code_format::ptx; }

backend_id cuda_source_object::managing_backend() const {
  return backend_id::cuda;
}

hcf_object_id cuda_source_object::hcf_source() const { return _origin; }

std::string cuda_source_object::target_arch() const { return _target_arch; }

compilation_flow cuda_source_object::source_compilation_flow() const {
  return compilation_flow::explicit_multipass;
}

std::vector<std::string>
cuda_source_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

bool cuda_source_object::contains(
    const std::string &backend_kernel_name) const {
  // TODO We cannot use proper equality checks because the kernel prefix
  // might vary depending on the clang version
  for (const auto &name : _kernel_names) {
    if (name.find(backend_kernel_name) != std::string::npos)
      return true;
  }
  return false;
}

const std::string &cuda_source_object::get_source() const { return _source; }

cuda_multipass_executable_object::cuda_multipass_executable_object(
    const cuda_source_object *source, int device)
    : _source{source}, _device{device}, _module{nullptr} {

  assert(source);

  this->_build_result = build();
}

result cuda_multipass_executable_object::get_build_result() const {
  return _build_result;
}

cuda_multipass_executable_object::~cuda_multipass_executable_object() {
  unload_cuda_module(_module, _device);
}

code_object_state cuda_multipass_executable_object::state() const {
  return _module ? code_object_state::executable : code_object_state::invalid;
}

code_format cuda_multipass_executable_object::format() const {
  return code_format::ptx;
}

backend_id cuda_multipass_executable_object::managing_backend() const {
  return backend_id::cuda;
}

hcf_object_id cuda_multipass_executable_object::hcf_source() const {
  return _source->hcf_source();
}

std::string cuda_multipass_executable_object::target_arch() const {
  return _source->target_arch();
}

compilation_flow
cuda_multipass_executable_object::source_compilation_flow() const {
  return compilation_flow::explicit_multipass;
}

std::vector<std::string>
cuda_multipass_executable_object::supported_backend_kernel_names() const {
  return _source->supported_backend_kernel_names();
}

bool cuda_multipass_executable_object::contains(
    const std::string &backend_kernel_name) const {
  return _source->contains(backend_kernel_name);
}

CUmod_st* cuda_multipass_executable_object::get_module() const {
  return _module;
}

result cuda_multipass_executable_object::build() {
  if (_module != nullptr)
    return make_success();

  return build_cuda_module_from_ptx(_module, _device, _source->get_source());
}

int cuda_multipass_executable_object::get_device() const {
  return _device;
}

cuda_sscp_executable_object::cuda_sscp_executable_object(
    const std::string &ptx_source, const std::string &target_arch,
    hcf_object_id hcf_source, const std::vector<std::string> &kernel_names,
    int device, const glue::kernel_configuration &config)
    : _target_arch{target_arch}, _hcf{hcf_source}, _kernel_names{kernel_names},
      _device{device}, _id{config.generate_id()}, _module{nullptr} {
  _build_result = build(ptx_source);
}

cuda_sscp_executable_object::~cuda_sscp_executable_object() {
  unload_cuda_module(_module, _device);
}

result cuda_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state cuda_sscp_executable_object::state() const {
  return _module ? code_object_state::executable : code_object_state::invalid;
}

code_format cuda_sscp_executable_object::format() const {
  return code_format::ptx;
}

backend_id cuda_sscp_executable_object::managing_backend() const {
  return backend_id::cuda;
}

hcf_object_id cuda_sscp_executable_object::hcf_source() const {
  return _hcf;
}

std::string cuda_sscp_executable_object::target_arch() const {
  return _target_arch;
}

compilation_flow cuda_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
cuda_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

CUmod_st* cuda_sscp_executable_object::get_module() const {
  return _module;
}

int cuda_sscp_executable_object::get_device() const {
  return _device;
}

result cuda_sscp_executable_object::build(const std::string& source) {
  if (_module != nullptr)
    return make_success();

  return build_cuda_module_from_ptx(_module, _device, source);
}

bool cuda_sscp_executable_object::contains(const std::string &backend_kernel_name) const {
  for(const auto& kernel_name : _kernel_names) {
    if(kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

} // namespace rt
} // namespace hipsycl
