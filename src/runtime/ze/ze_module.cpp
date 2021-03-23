/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#include "hipSYCL/runtime/ze/ze_module.hpp"
#include "hipSYCL/runtime/ze/ze_event.hpp"
#include "hipSYCL/runtime/ze/ze_hardware_manager.hpp"
#include "hipSYCL/runtime/ze/ze_queue.hpp"
#include "hipSYCL/runtime/error.hpp"
#include <bits/stdint-uintn.h>
#include <level_zero/ze_api.h>
#include <cassert>

namespace hipsycl {
namespace rt {

result ze_module_invoker::submit_kernel(
    module_id_t id, const std::string &module_variant,
    const std::string *module_image, const rt::range<3> &num_groups,
    const rt::range<3> &group_size, unsigned local_mem_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args,
    const std::string &kernel_name_tag, const std::string &kernel_body_name) {
  
  assert(module_image);

  ze_hardware_manager* hw_mgr = _queue->get_hardware_manager();
  assert(hw_mgr);
  ze_hardware_context* hw_ctx =
      static_cast<ze_hardware_context *>(
          hw_mgr->get_device(_queue->get_device().get_id()));
  
  assert(hw_ctx);
  ze_module *mod = nullptr;
  result res = hw_ctx->obtain_module(id, module_variant, module_image, mod);
  if(!res.is_success())
    return res;

  if(!mod) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ze_module_invoker: Could not obtain module for kernel image"});
  }

  if(!mod->get_build_status().is_success()) {
    return make_error(
        __hipsycl_here(),
        error_info{
            "ze_module_invoker: Module construction failed"});
  }


  ze_kernel_handle_t kernel;
  res = mod->obtain_kernel(kernel_name_tag, kernel_body_name, kernel);

  if(!res.is_success())
    return res;

  ze_result_t err =
      zeKernelSetGroupSize(kernel, static_cast<uint32_t>(group_size[0]),
                           static_cast<uint32_t>(group_size[1]),
                           static_cast<uint32_t>(group_size[2]));
  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_module_invoker: Could not set kernel group size",
                   error_code{"ze", static_cast<int>(err)}});
  }

  ze_group_count_t group_count;
  group_count.groupCountX = static_cast<uint32_t>(num_groups[0]);
  group_count.groupCountY = static_cast<uint32_t>(num_groups[1]);
  group_count.groupCountZ = static_cast<uint32_t>(num_groups[2]);

  for(std::size_t i = 0; i < num_args; ++i ){
    HIPSYCL_DEBUG_INFO << "ze_module_invoker: Setting kernel argument " << i
                       << " of size " << arg_sizes[i] << " at " << args[i]
                       << std::endl;

    err = zeKernelSetArgumentValue(
        kernel, i, static_cast<uint32_t>(arg_sizes[i]), args[i]);
    if(err != ZE_RESULT_SUCCESS) {
      return make_error(
          __hipsycl_here(),
          error_info{"ze_module_invoker: Could not set kernel argument",
                     error_code{"ze", static_cast<int>(err)}});
    }
  }

  // This is necessary for USM pointers, which hipSYCL *always*
  // relies on.
  err = zeKernelSetIndirectAccess(kernel,
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_HOST |
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_DEVICE |
                                  ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED);

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
          __hipsycl_here(),
          error_info{"ze_module_invoker: Could not set indirect access flags",
                     error_code{"ze", static_cast<int>(err)}});
  }

  std::vector<ze_event_handle_t> wait_events =
      _queue->get_enqueued_event_handles();
  std::shared_ptr<dag_node_event> completion_evt = _queue->create_event();

  HIPSYCL_DEBUG_INFO << "ze_module_invoker: Submitting kernel!" << std::endl;
  err = zeCommandListAppendLaunchKernel(
      _queue->get_ze_command_list(), kernel, &group_count,
      static_cast<ze_node_event *>(completion_evt.get())->get_event_handle(),
      static_cast<uint32_t>(wait_events.size()), wait_events.data());
  
  if(err != ZE_RESULT_SUCCESS) {
    return make_error(
        __hipsycl_here(),
        error_info{"ze_module_invoker: Kernel launch failed",
                   error_code{"ze", static_cast<int>(err)}});
  }

  _queue->register_submitted_op(completion_evt);

  return make_success();
}

ze_module::ze_module(ze_context_handle_t ctx, ze_device_handle_t dev, module_id_t id,
            const std::string& variant, const std::string *module_image)
            : _id{id}, _variant{variant}, _dev{dev} {

  assert(module_image);

  ze_module_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_MODULE_DESC;
  desc.pNext = nullptr;
  
  if(variant == "spirv") {
    desc.format = ZE_MODULE_FORMAT_IL_SPIRV;
  } else {
    _build_status = register_error(
        __hipsycl_here(), error_info{"ze_module: Invalid module format"});
    return;
  }
  
  desc.inputSize = module_image->size();
  desc.pInputModule = reinterpret_cast<const uint8_t *>(module_image->c_str());
  // TODO: We may want to expose some of the build flags, e.g. to
  // enable greater than 4GB buffers
  desc.pBuildFlags = nullptr;
  desc.pConstants = nullptr;

  ze_result_t err = zeModuleCreate(ctx, dev, &desc, &_handle, nullptr);

  if(err != ZE_RESULT_SUCCESS) {
    _build_status = register_error(__hipsycl_here(),
                   error_info{"ze_module: Couldn't create module handle",
                              error_code{"ze", static_cast<int>(err)}});
  } else {
    _build_status = make_success();
  }
}

ze_module::~ze_module() {
  ze_result_t err = zeModuleDestroy(_handle);
  if(err != ZE_RESULT_SUCCESS) {
    register_error(__hipsycl_here(),
                   error_info{"ze_module: Couldn't destroy module handle",
                              error_code{"ze", static_cast<int>(err)}});
  }
}
  
ze_module_handle_t ze_module::get_handle() const {
  return _handle;
}

module_id_t ze_module::get_id() const {
  return _id;
}

ze_device_handle_t ze_module::get_device() const {
  return _dev;
}

result ze_module::get_build_status() const {
  return _build_status;
}

const std::string& ze_module::get_variant() const {
  return _variant;
}

result ze_module::obtain_kernel(const std::string &name,
                                ze_kernel_handle_t &out) const {
  for(auto& kernel : _kernels) {
    if (kernel.first == name) {
      HIPSYCL_DEBUG_INFO << "ze_module: Found cached kernel: " << name
                         << std::endl;

      out = kernel.second;
      return make_success();
    }
  }

  ze_kernel_desc_t desc;
  desc.stype = ZE_STRUCTURE_TYPE_KERNEL_DESC;
  desc.pNext = nullptr;
  desc.flags = 0;
  desc.pKernelName = name.c_str();

  ze_kernel_handle_t kernel;
  ze_result_t err = zeKernelCreate(_handle, &desc, &kernel);

  if(err != ZE_RESULT_SUCCESS) {
    return make_error(__hipsycl_here(),
                      error_info{"ze_module: Couldn't construct kernel",
                                 error_code{"ze", static_cast<int>(err)}});
  }
  _kernels.emplace_back(std::make_pair(name, kernel));
  out = kernel;
  HIPSYCL_DEBUG_INFO << "ze_module: Constructed new kernel for cache: " << name
                     << std::endl;

  return make_success();
}

result ze_module::obtain_kernel(const std::string &name,
                                const std::string &fallback_name,
                                ze_kernel_handle_t &out) const {
  for(auto& kernel : _kernels) {
    if (kernel.first == name || kernel.first == fallback_name) {
      out = kernel.second;
      return make_success();
    }
  }

  result res = obtain_kernel(name, out);
  if(!res.is_success()) {
    res = obtain_kernel(fallback_name, out);
    if(!res.is_success()) {
      return make_error(__hipsycl_here(),
                        error_info{"ze_module: Couldn't construct kernel "
                                   "neither with name, nor fallback name."});
    }
  }

  return make_success();
}

}
}
