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
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cassert>

#include <cuda_runtime_api.h>
#include <cuda.h>

#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
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
          __acpp_here(),
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

  static constexpr std::size_t num_options = 2;
  std::array<CUjit_option, num_options> option_names{};
  std::array<void*, num_options> option_vals{};

  // set up size of compilation log buffer
  option_names[0] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;
  static constexpr std::size_t error_log_buffer_size = 10*1024;
  option_vals[0] = reinterpret_cast<void*>(error_log_buffer_size);

  // set up pointer to the compilation log buffer
  option_names[1] = CU_JIT_ERROR_LOG_BUFFER;
  std::string error_log_buffer(error_log_buffer_size, '\0');
  option_vals[1] = error_log_buffer.data();

  auto err = cuModuleLoadDataEx(
      &module, source.data(),
      num_options, option_names.data(), option_vals.data());

  if (err != CUDA_SUCCESS) {
    const auto error_log_size = reinterpret_cast<std::size_t>(option_vals[0]);
    error_log_buffer.resize(error_log_size);
    return make_error(
        __acpp_here(),
        error_info{
            "cuda_executable_object: Could not load module, CUDA JIT log: " +
                error_log_buffer,
            error_code{"CU", static_cast<int>(err)}});
  }

  assert(module);

  return make_success();
}

std::vector<std::string> extract_kernel_names_from_ptx(const std::string& source) {

  std::vector<std::string> kernel_names;
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
      kernel_names.push_back(line);
    }
  }

  return kernel_names;
}

}


cuda_multipass_executable_object::cuda_multipass_executable_object(hcf_object_id origin,
                                   const std::string &target,
                                   const std::string &source, int device)
    : _origin{origin}, _target{target}, _device{device}, _module{nullptr} {

  this->_build_result = build(source);
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
  return _origin;
}

std::string cuda_multipass_executable_object::target_arch() const {
  return _target;
}

compilation_flow
cuda_multipass_executable_object::source_compilation_flow() const {
  return compilation_flow::explicit_multipass;
}

std::vector<std::string>
cuda_multipass_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

bool cuda_multipass_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for(const auto& k : _kernel_names)
    if(k == backend_kernel_name)
      return true;
  return false;
}

CUmod_st* cuda_multipass_executable_object::get_module() const {
  return _module;
}

result cuda_multipass_executable_object::build(const std::string& source) {
  if (_module != nullptr)
    return make_success();

  _kernel_names = extract_kernel_names_from_ptx(source);
  return build_cuda_module_from_ptx(_module, _device, source);
}

int cuda_multipass_executable_object::get_device() const {
  return _device;
}

cuda_sscp_executable_object::cuda_sscp_executable_object(
    const std::string &ptx_source, const std::string &target_arch,
    hcf_object_id hcf_source, const std::vector<std::string> &kernel_names,
    int device, const kernel_configuration &config)
    : _target_arch{target_arch}, _hcf{hcf_source}, _kernel_names{kernel_names},
      _id{config.generate_id()}, _device{device}, _module{nullptr} {
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
