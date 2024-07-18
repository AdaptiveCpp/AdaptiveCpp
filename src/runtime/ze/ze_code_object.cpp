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
#include "hipSYCL/runtime/ze/ze_code_object.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/ze/ze_event.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <algorithm>
#include <bits/stdint-uintn.h>
#include <level_zero/ze_api.h>
#include <cassert>
#include <string>
#include <vector>

namespace hipsycl {
namespace rt {

result ze_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned int local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, std::string_view kernel_name,
    const rt::hcf_kernel_info *kernel_info,
    const kernel_configuration &config) {

  assert(_queue);

  return _queue->submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, kernel_info, num_groups, group_size,
      local_mem_size, args, arg_sizes, num_args, config);
}

ze_executable_object::ze_executable_object(ze_context_handle_t ctx,
                                           ze_device_handle_t dev,
                                           hcf_object_id source,
                                           ze_source_format fmt,
                                           const std::string &code_image)
    : _source{source}, _format{fmt}, _ctx{ctx}, _dev{dev}, _module{nullptr}
{

  ze_module_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.pNext = nullptr;
  
  if(_format == ze_source_format::spirv) {
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  } else {
    _build_status = register_error(
        __acpp_here(), error_info{"ze_executable_object: Invalid module format"});
    return;
  }
  
  desc.inputSize = code_image.size();
  desc.pInputModule = reinterpret_cast<const uint8_t *>(code_image.c_str());
  // TODO: We may want to expose some of the build flags, e.g. to
  // enable greater than 4GB buffers
  desc.pBuildFlags = nullptr;
  desc.pConstants = nullptr;

  ze_module_build_log_handle_t build_log;
  ze_result_t err = zeModuleCreate(ctx, dev, &desc, &_module, &build_log);

  if(err != ZE_RESULT_SUCCESS) {
    std::size_t build_log_size;
    std::string build_log_content;

    if (zeModuleBuildLogGetString(build_log, &build_log_size, nullptr) ==
        ZE_RESULT_SUCCESS) {
      std::vector<char> build_log_buffer(build_log_size);
      if (zeModuleBuildLogGetString(build_log, &build_log_size,
                                    build_log_buffer.data()) ==
          ZE_RESULT_SUCCESS) {
        build_log_content = std::string{build_log_buffer.data(), build_log_buffer.size()};
      }
    }

    std::string msg = "ze_executable_object: Couldn't create module handle";
    if(!build_log_content.empty()) {
      msg += "\nBuild log: ";
      msg += build_log_content;
    }
    _build_status = register_error(__acpp_here(),
                   error_info{msg,
                              error_code{"ze", static_cast<int>(err)}});
    zeModuleBuildLogDestroy(build_log);
    return;
  } else {
    zeModuleBuildLogDestroy(build_log);
    _build_status = make_success();
  }

  HIPSYCL_DEBUG_INFO << "ze_executable_object: Successfully created module "
                        "from code image of size "
                     << code_image.size() << std::endl;

  uint32_t num_kernels = 0;
  err = zeModuleGetKernelNames(_module, &num_kernels, nullptr);
  if (err != ZE_RESULT_SUCCESS) {
    register_error(
        __acpp_here(),
        error_info{"ze_executable_object: Couldn't obtain number of kernels",
                   error_code{"ze", static_cast<int>(err)}});
    return;
  }

  std::vector<const char*> kernel_names(num_kernels, nullptr);
  err = zeModuleGetKernelNames(_module, &num_kernels, kernel_names.data());

  if (err != ZE_RESULT_SUCCESS) {
    register_error(
        __acpp_here(),
        error_info{"ze_executable_object: Couldn't obtain kernel names",
                   error_code{"ze", static_cast<int>(err)}});
    return;
  }

  for(const char* name : kernel_names)
    _kernels.push_back(std::string{name});

  load_kernel_handles();  
}

ze_executable_object::~ze_executable_object() {
  if(_module) {
    ze_result_t err = zeModuleDestroy(_module);
    if(err != ZE_RESULT_SUCCESS) {
      register_error(__acpp_here(),
                   error_info{"ze_executable_object: Couldn't destroy module handle",
                              error_code{"ze", static_cast<int>(err)}});
    }
  }
}

result ze_executable_object::get_build_result() const{
  return _build_status;
}

code_object_state ze_executable_object::state() const {
  return _module ? code_object_state::executable : code_object_state::source;
}

code_format ze_executable_object::format() const {
  if(_format == ze_source_format::spirv)
    return code_format::spirv;
  else
    return code_format::native_isa;
}

backend_id ze_executable_object::managing_backend() const {
  return backend_id::level_zero;
}

hcf_object_id ze_executable_object::hcf_source() const {
  return _source;
}

std::string ze_executable_object::target_arch() const {
  // TODO might want to return actual device name that we have compiled for?
  return "spirv64";
}

compilation_flow ze_executable_object::source_compilation_flow() const {
  return compilation_flow::explicit_multipass;
}

std::vector<std::string> ze_executable_object::supported_backend_kernel_names( ) const {
  return _kernels;
}

bool ze_executable_object::contains(const std::string &backend_kernel_name) const {
  return std::find(_kernels.begin(), _kernels.end(), backend_kernel_name) !=
         _kernels.end();
}

ze_device_handle_t ze_executable_object::get_ze_device() const {
  return _dev;
}

ze_context_handle_t ze_executable_object::get_ze_context() const {
  return _ctx;
}

result ze_executable_object::get_kernel(std::string_view kernel_name,
                                        ze_kernel_handle_t &out) const {
  assert(_module);

  auto k = _kernel_handles.find(kernel_name);
  if(k != _kernel_handles.end()) {
    out = k->second;
    HIPSYCL_DEBUG_INFO << "ze_executable_object: Found cached kernel: " << kernel_name
                    << std::endl;
    return make_success();
  }

  return make_error(__acpp_here(),
                    error_info{"ze_executable_object: get_kernel(): Attempted "
                               "to access kernels that is unavailable"});
}

void ze_executable_object::load_kernel_handles() {
  for(const auto& kernel_name : _kernels) {
    ze_kernel_desc_t desc;
    desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
    desc.pNext = nullptr;
    desc.flags = 0;
    desc.pKernelName = kernel_name.c_str();

    ze_kernel_handle_t kernel;
    ze_result_t err = zeKernelCreate(_module, &desc, &kernel);

    if(err == ZE_RESULT_SUCCESS) {
      _kernel_handles[kernel_name] = kernel;
    
      HIPSYCL_DEBUG_INFO
          << "ze_executable_object: Constructed new kernel for cache: "
          << kernel_name << std::endl;
    }
  }
}


ze_sscp_executable_object::ze_sscp_executable_object(ze_context_handle_t ctx, ze_device_handle_t dev,
                          hcf_object_id source,
                          const std::string &spirv_image,
                          const kernel_configuration &config)
    : ze_executable_object(ctx, dev, source, ze_source_format::spirv,
                            spirv_image),
      _id{config.generate_id()} {}


compilation_flow ze_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

kernel_configuration::id_type ze_sscp_executable_object::configuration_id() const{
  return _id;
}



}
}
