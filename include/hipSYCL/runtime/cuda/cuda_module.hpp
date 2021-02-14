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

#ifndef HIPSYCL_CUDA_MODULE_HPP
#define HIPSYCL_CUDA_MODULE_HPP

#include <vector>
#include <string>

#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/module_invoker.hpp"
#include "hipSYCL/glue/generic/module.hpp"

struct CUmod_st;

namespace hipsycl {
namespace rt {

using cuda_module_id_t = module_id_t;
class cuda_queue;

class cuda_module {
public:
  cuda_module(cuda_module_id_t module_id, const std::string &target,
              const std::string &code_content);
  
  const std::vector<std::string>& get_kernel_names() const;

  std::string get_content() const;

  bool guess_kernel_name(const std::string &kernel_group_name,
                         const std::string &kernel_component_name,
                         std::string &guessed_name) const;

  cuda_module_id_t get_id() const;
  const std::string& get_target() const;
  
private:
  cuda_module_id_t _id;
  std::string _target;
  std::string _content;
  std::vector<std::string> _kernel_names;
  
};

class cuda_module_manager {
public:
  cuda_module_manager() = default;
  cuda_module_manager(std::size_t num_devices);
  ~cuda_module_manager();

  const cuda_module &obtain_module(cuda_module_id_t id,
                                   const std::string &target,
                                   const std::string &content);

  result load(rt::device_id dev, const cuda_module &module, CUmod_st*& out);

private:
  std::size_t _num_devices;

  // Cache constructed modules
  std::vector<cuda_module> _modules;

  // Store active CUDA module per device
  std::vector<CUmod_st *> _cuda_modules;
  std::vector<cuda_module_id_t> _active_modules;
};

}
}

#endif
