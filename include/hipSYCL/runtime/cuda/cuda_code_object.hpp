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

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

struct CUmod_st;

namespace hipsycl {
namespace rt {

class cuda_source_object : public code_object {
public:
  virtual ~cuda_source_object(){}
  cuda_source_object(hcf_object_id origin, const std::string &target,
                     const std::string &source);

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;

  virtual bool contains(const std::string &backend_kernel_name) const override;
  const std::string& get_source() const;

private:
  hcf_object_id _origin;
  std::vector<std::string> _kernel_names;
  std::string _target_arch;
  std::string _source;
};

class cuda_executable_object : public code_object {
public:
  virtual ~cuda_executable_object();
  cuda_executable_object(const cuda_source_object* source, int device);

  result get_build_result() const;

  virtual code_object_state state() const override;
  virtual code_format format() const override;
  virtual backend_id managing_backend() const override;
  virtual hcf_object_id hcf_source() const override;
  virtual std::string target_arch() const override;

  virtual std::vector<std::string>
  supported_backend_kernel_names() const override;
  virtual bool contains(const std::string &backend_kernel_name) const override;

  CUmod_st* get_module() const;
  int get_device() const;
private:
  result build();

  result _build_result;
  const cuda_source_object* _source;
  int _device;
  CUmod_st* _module;
};


}
}

#endif
