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
#ifndef HIPSYCL_OCL_CODE_OBJECT_HPP
#define HIPSYCL_OCL_CODE_OBJECT_HPP

#include <string>

#include <CL/opencl.hpp>

#include "hipSYCL/runtime/code_object_invoker.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"

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
                               std::string_view kernel_name,
                               const rt::hcf_kernel_info* kernel_info,
                               const kernel_configuration& config) override;
private:
  ocl_queue* _queue;
};


class ocl_executable_object : public code_object {
public:
  ocl_executable_object(const cl::Context &ctx, cl::Device &dev,
                        hcf_object_id source, const std::string &code_image,
                        const kernel_configuration &config);
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
  virtual kernel_configuration::id_type configuration_id() const override;

  cl::Device get_cl_device() const;
  cl::Context get_cl_context() const;

  // Only works if the module has been built successfully
  result get_kernel(std::string_view name, cl::Kernel& out) const;
private:
  hcf_object_id _source;
  cl::Context _ctx;
  cl::Device _dev;
  cl::Program _program;
  
  std::vector<std::string> _kernel_names;
  std::unordered_map<std::string_view, cl::Kernel> _kernel_handles;

  result _build_status;
  kernel_configuration::id_type _id;
};


}
}

#endif
