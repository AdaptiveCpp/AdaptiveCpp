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
#ifndef HIPSYCL_HIP_CODE_OBJECT_HPP
#define HIPSYCL_HIP_CODE_OBJECT_HPP

#include <vector>
#include <string>

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"


struct ihipModule_t;

namespace hipsycl {
namespace rt {

class hip_executable_object : public code_object {
public:
  virtual ~hip_executable_object() {}
  virtual ihipModule_t* get_module() const = 0;
  virtual result get_build_result() const = 0;
  virtual int get_device() const = 0;
};

class hip_multipass_executable_object : public hip_executable_object {
public:
  virtual ~hip_multipass_executable_object();
  hip_multipass_executable_object(hcf_object_id origin, const std::string &target,
                        const std::string &hip_fat_binary, int device);

  virtual result get_build_result() const override;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;
  virtual compilation_flow source_compilation_flow() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;

  virtual bool contains(const std::string &backend_kernel_name) const override;

  virtual ihipModule_t* get_module() const override;
  virtual int get_device() const override;
private:
  result build(const std::string& hip_fat_binary);

  hcf_object_id _origin;
  std::string _target;
  result _build_result;
  int _device;
  ihipModule_t* _module;
};

class hip_sscp_executable_object : public hip_executable_object {
public:
  virtual ~hip_sscp_executable_object();
  hip_sscp_executable_object(const std::string &hip_fat_binary,
                             const std::string &target_arch,
                             hcf_object_id source,
                             const std::vector<std::string> &kernel_name,
                             int device,
                             const kernel_configuration &config);

  virtual result get_build_result() const override;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;
  virtual compilation_flow source_compilation_flow() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;

  virtual bool contains(const std::string &backend_kernel_name) const override;

  virtual ihipModule_t* get_module() const override;
  virtual int get_device() const override;
private:
  result build(const std::string& hip_fat_binary);

  std::string _target;
  hcf_object_id _origin;
  std::vector<std::string> _kernel_names;
  result _build_result;
  kernel_configuration::id_type _id;
  int _device;
  ihipModule_t* _module;
};


}
}

#endif
