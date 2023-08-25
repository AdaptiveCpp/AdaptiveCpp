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

#ifndef HIPSYCL_OCL_CODE_OBJECT_HPP
#define HIPSYCL_OCL_CODE_OBJECT_HPP

#include <string>

#include <CL/opencl.hpp>

#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"

namespace hipsycl {
namespace rt {

class ocl_queue;

class ocl_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  ocl_sscp_code_object_invoker(ocl_queue* queue)
  : _queue{queue} {}

  virtual ~ocl_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name,
                               const glue::kernel_configuration& config) override;
private:
  ocl_queue* _queue;
};


class ocl_executable_object : public code_object {
public:
  ocl_executable_object(const cl::Context& ctx, cl::Device& dev,
    hcf_object_id source, const std::string& code_image, const glue::kernel_configuration &config);
  virtual ~ocl_executable_object();

  result get_build_result() const;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;
  virtual bool contains(const std::string &backend_kernel_name) const override;

  virtual compilation_flow source_compilation_flow() const override;
  virtual glue::kernel_configuration::id_type configuration_id() const override;

  cl::Device get_cl_device() const;
  cl::Context get_cl_context() const;

  // Only works if the module has been built successfully
  result get_kernel(const std::string& name, cl::Kernel& out) const;
private:
  hcf_object_id _source;
  cl::Context _ctx;
  cl::Device _dev;
  cl::Program _program;
  
  mutable std::unordered_map<std::string, cl::Kernel> _kernel_handles;

  result _build_status;
  glue::kernel_configuration::id_type _id;
};


}
}

#endif
