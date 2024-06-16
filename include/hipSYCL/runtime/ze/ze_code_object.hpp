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

#ifndef HIPSYCL_ZE_CODE_OBJECT_HPP
#define HIPSYCL_ZE_CODE_OBJECT_HPP

#include <string>

#include <level_zero/ze_api.h>

#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"

namespace hipsycl {
namespace rt {

class ze_queue;


class ze_sscp_code_object_invoker : public sscp_code_object_invoker {
public:
  ze_sscp_code_object_invoker(ze_queue* queue)
  : _queue{queue} {}

  virtual ~ze_sscp_code_object_invoker(){}

  virtual result submit_kernel(const kernel_operation& op,
                               hcf_object_id hcf_object,
                               const rt::range<3> &num_groups,
                               const rt::range<3> &group_size,
                               unsigned local_mem_size, void **args,
                               std::size_t *arg_sizes, std::size_t num_args,
                               const std::string &kernel_name,
                               const kernel_configuration& config) override;
private:
  ze_queue* _queue;
};

enum class ze_source_format {
  spirv,
  native
};

class ze_executable_object : public code_object {
public:
  ze_executable_object(ze_context_handle_t ctx, ze_device_handle_t dev,
    hcf_object_id source, ze_source_format fmt, const std::string& code_image);
  virtual ~ze_executable_object();

  result get_build_result() const;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;
  virtual compilation_flow source_compilation_flow() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;
  virtual bool contains(const std::string &backend_kernel_name) const override;

  ze_device_handle_t get_ze_device() const;
  ze_context_handle_t get_ze_context() const;
  // This should only be called inside ze_queue, not the user,
  // so we do not have to worry about thread-safety. Only works
  // if the module has been built successfully
  result get_kernel(const std::string& name, ze_kernel_handle_t& out) const;
private:
  ze_source_format _format;
  hcf_object_id _source;
  ze_context_handle_t _ctx;
  ze_device_handle_t _dev;
  ze_module_handle_t _module;
  std::vector<std::string> _kernels;
  mutable std::unordered_map<std::string, ze_kernel_handle_t> _kernel_handles;

  result _build_status;
};

class ze_sscp_executable_object : public ze_executable_object {
public:
  ze_sscp_executable_object(ze_context_handle_t ctx, ze_device_handle_t dev,
                            hcf_object_id source,
                            const std::string &spirv_image,
                            const kernel_configuration &config);
  ~ze_sscp_executable_object() {}

  virtual compilation_flow source_compilation_flow() const override;
  virtual kernel_configuration::id_type configuration_id() const override;
private:
  kernel_configuration::id_type _id;
};

}
}

#endif
