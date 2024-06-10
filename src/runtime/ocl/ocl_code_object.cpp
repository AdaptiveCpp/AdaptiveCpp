/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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

#include "hipSYCL/runtime/ocl/ocl_code_object.hpp"
#include "hipSYCL/common/string_utils.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/ocl/ocl_queue.hpp"

namespace hipsycl {
namespace rt {

result ocl_sscp_code_object_invoker::submit_kernel(
    const kernel_operation &op, hcf_object_id hcf_object,
    const rt::range<3> &num_groups, const rt::range<3> &group_size,
    unsigned int local_mem_size, void **args, std::size_t *arg_sizes,
    std::size_t num_args, const std::string &kernel_name,
    const kernel_configuration &config) {

  assert(_queue);

  return _queue->submit_sscp_kernel_from_code_object(
      op, hcf_object, kernel_name, num_groups, group_size, local_mem_size, args,
      arg_sizes, num_args, config);
}

ocl_executable_object::ocl_executable_object(const cl::Context& ctx, cl::Device& dev,
    hcf_object_id source, const std::string& code_image, const kernel_configuration &config)
: _source{source}, _ctx{ctx}, _dev{dev}, _id{config.generate_id()} {

  std::vector<char> ir(code_image.size());
  std::memcpy(ir.data(), code_image.data(), code_image.size());

  cl_int err = 0;
  _program = cl::Program(_ctx, ir, false, &err);

  if(err != CL_SUCCESS) {
    _build_status = register_error(
        __hipsycl_here(),
        error_info{"ocl_code_object: Construction of CL program failed",
                   error_code{"CL", static_cast<int>(err)}});
    return;
  }
  
  std::string options_string="-cl-uniform-work-group-size";
  for(const auto& flag : config.build_flags()) {
    if(flag == kernel_build_flag::fast_math) {
      options_string += " -cl-fast-relaxed-math";
    }
  }

  err = _program.build(
      _dev, options_string.c_str());

  if(err != CL_SUCCESS) {
    std::string build_log = "<build log not available>";
    cl_int access_build_log_err =
        _program.getBuildInfo(_dev, CL_PROGRAM_BUILD_LOG, &build_log);

    std::string msg = "ocl_code_object: Building CL program failed.";
    if(access_build_log_err == CL_SUCCESS)
      msg += " Build log: " + build_log;
    
    _build_status = register_error(
        __hipsycl_here(), error_info{msg,
                                     error_code{"CL", static_cast<int>(err)}});
    return;
  }

  // clCreateKernelsInProgram seems to not work reliably
  //err = _program.createKernels(&kernels);
  std::string concatenated_name_list;
  err = _program.getInfo(CL_PROGRAM_KERNEL_NAMES, &concatenated_name_list);
  
  if(err != CL_SUCCESS) {
    _build_status = register_error(
        __hipsycl_here(),
        error_info{
            "ocl_code_object: Could not obtain kernel names in program",
            error_code{"CL", static_cast<int>(err)}});
    return;
  }

  std::vector<std::string> kernel_names =
      common::split_by_delimiter(concatenated_name_list, ';');
  std::vector<cl::Kernel> kernels;
  for(const auto& name : kernel_names) {
    cl::Kernel k{_program, name.c_str(), &err};
    if(err != CL_SUCCESS) {
      _build_status = register_error(
        __hipsycl_here(),
        error_info{
            "ocl_code_object: Could not construct kernel object for kernel "+name,
            error_code{"CL", static_cast<int>(err)}});
      return;
    }
    _kernel_handles[name] = k;
  }

  _build_status = make_success();
}

ocl_executable_object::~ocl_executable_object() {}

result ocl_executable_object::get_build_result() const {
  return _build_status;
}

code_object_state ocl_executable_object::state() const {
  if(_build_status.is_success())
    return code_object_state::executable;
  return code_object_state::invalid;
}

code_format ocl_executable_object::format() const {
  return code_format::spirv;
}

backend_id ocl_executable_object::managing_backend() const {
  return backend_id::ocl;
}

hcf_object_id ocl_executable_object::hcf_source() const {
  return _source;
}

std::string ocl_executable_object::target_arch() const {
  // TODO might want to return actual device name that we have compiled for?
  return "spirv64";
}

std::vector<std::string>
ocl_executable_object::supported_backend_kernel_names() const {
  std::vector<std::string> result;
  result.reserve(_kernel_handles.size());
  for(const auto& h : _kernel_handles)
    result.push_back(h.first);
  return result;
}

bool ocl_executable_object::contains(
    const std::string &backend_kernel_name) const {
  return _kernel_handles.find(backend_kernel_name) != _kernel_handles.end();
}

compilation_flow ocl_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

kernel_configuration::id_type
ocl_executable_object::configuration_id() const {
  return _id;
}

cl::Device ocl_executable_object::get_cl_device() const {
  return _dev;
}

cl::Context ocl_executable_object::get_cl_context() const {
  return _ctx;
}
  
result ocl_executable_object::get_kernel(const std::string& name, cl::Kernel& out) const {
  if(!_build_status.is_success())
    return _build_status;
  auto it = _kernel_handles.find(name);
  if(it == _kernel_handles.end())
    return make_error(__hipsycl_here(),
                      error_info{"ocl_executable_object: Unknown kernel name"});
  out = it->second;
  return make_success();
}

}
}
