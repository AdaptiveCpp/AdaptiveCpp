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
#include "hipSYCL/runtime/metal/metal_code_object.hpp"

#include <Metal/Metal.hpp>

#undef nil

#include "hipSYCL/common/debug.hpp"

namespace hipsycl {
namespace rt {

namespace {

result build_metal_library_from_source(MTL::Library*& library,
                                       MTL::Device* device,
                                       const std::string& source) {
  if (!device) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Device is null"});
  }

  NS::Error* error = nullptr;

  NS::SharedPtr<MTL::CompileOptions> options = NS::TransferPtr(MTL::CompileOptions::alloc()->init());
  options->setLanguageVersion(MTL::LanguageVersion4_0);
  options->setOptimizationLevel(MTL::LibraryOptimizationLevel::LibraryOptimizationLevelSize);

  NS::String* sourceString = NS::String::string(source.c_str(),
                                                NS::UTF8StringEncoding);

  library = device->newLibrary(sourceString, options.get(), &error);

  if (error) {
    std::string error_msg = "metal_code_object: Shader compilation failed";
    if (error->localizedDescription()) {
      error_msg += ": ";
      error_msg += error->localizedDescription()->utf8String();
    }
    return make_error(__acpp_here(), error_info{error_msg});
  }

  if (!library) {
    return make_error(__acpp_here(),
                      error_info{"metal_code_object: Library creation failed "
                                 "without error message"});
  }

  HIPSYCL_DEBUG_INFO << "metal_code_object: Successfully compiled Metal shader"
                     << std::endl;

  return make_success();
}

} // anonymous namespace

metal_sscp_executable_object::metal_sscp_executable_object(
    const std::string &metal_source, const std::string &target_arch,
    hcf_object_id hcf_source, const std::vector<std::string> &kernel_names,
    MTL::Device* device, const kernel_configuration &config)
    : _target_arch{target_arch}, _hcf{hcf_source}, _kernel_names{kernel_names},
      _id{config.generate_id()}, _device{device}, _library{nullptr},
      _msl_source{metal_source} {
  _build_result = build(metal_source);
}

metal_sscp_executable_object::~metal_sscp_executable_object() {
  if (_library) {
    _library->release();
  }
}

result metal_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state metal_sscp_executable_object::state() const {
  return _library ? code_object_state::executable : code_object_state::invalid;
}

code_format metal_sscp_executable_object::format() const {
  // Metal uses its own shading language format
  return code_format::native_isa;
}

backend_id metal_sscp_executable_object::managing_backend() const {
  return backend_id::metal;
}

hcf_object_id metal_sscp_executable_object::hcf_source() const {
  return _hcf;
}

std::string metal_sscp_executable_object::target_arch() const {
  return _target_arch;
}

compilation_flow metal_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
metal_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

MTL::Library* metal_sscp_executable_object::get_library() const {
  return _library;
}

MTL::Device* metal_sscp_executable_object::get_device() const {
  return _device;
}

result metal_sscp_executable_object::build(const std::string& source) {
  if (_library != nullptr)
    return make_success();

  return build_metal_library_from_source(_library, _device, source);
}

bool metal_sscp_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for (const auto& kernel_name : _kernel_names) {
    if (kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

} // namespace rt
} // namespace hipsycl
