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
                               std::string_view kernel_name,
                               const rt::hcf_kernel_info* kernel_info,
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
  result get_kernel(std::string_view name, ze_kernel_handle_t& out) const;
private:
  ze_source_format _format;
  hcf_object_id _source;
  ze_context_handle_t _ctx;
  ze_device_handle_t _dev;
  ze_module_handle_t _module;
  std::vector<std::string> _kernels;
  std::unordered_map<std::string_view, ze_kernel_handle_t> _kernel_handles;
  
  void load_kernel_handles();

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
