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

#ifndef HIPSYCL_CUDA_CODE_OBJECT_HPP
#define HIPSYCL_CUDA_CODE_OBJECT_HPP

#include <vector>
#include <string>

#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

struct CUmod_st;

namespace hipsycl {
namespace rt {


class cuda_executable_object : public code_object {
public:
  virtual ~cuda_executable_object() {}
  virtual CUmod_st* get_module() const = 0;
  virtual result get_build_result() const = 0;
  virtual int get_device() const = 0;
};

class cuda_multipass_executable_object : public cuda_executable_object {
public:
  virtual ~cuda_multipass_executable_object();
  cuda_multipass_executable_object(hcf_object_id origin,
                                   const std::string &target,
                                   const std::string &source, int device);

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

  virtual CUmod_st* get_module() const override;
  virtual int get_device() const override;
private:
  hcf_object_id _origin;
  std::string _target;

  result build(const std::string& source);

  result _build_result;
  int _device;
  CUmod_st* _module;
  std::vector<std::string> _kernel_names;
};

class cuda_sscp_executable_object : public cuda_executable_object {
public:
  cuda_sscp_executable_object(const std::string &ptx_source,
                              const std::string &target_arch,
                              hcf_object_id hcf_source,
                              const std::vector<std::string> &kernel_names,
                              int device,
                              const kernel_configuration &config);

  virtual ~cuda_sscp_executable_object();

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

  virtual CUmod_st* get_module() const override;
  virtual int get_device() const override;
private:
  result build(const std::string& source);

  std::string _target_arch;
  hcf_object_id _hcf;
  std::vector<std::string> _kernel_names;
  result _build_result;
  kernel_configuration::id_type _id;
  int _device;
  CUmod_st* _module;
};

}
}

#endif
