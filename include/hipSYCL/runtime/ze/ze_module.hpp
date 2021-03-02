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

#ifndef HIPSYCL_ZE_MODULE_HPP
#define HIPSYCL_ZE_MODULE_HPP

#include <string>

#include <level_zero/ze_api.h>

#include "hipSYCL/runtime/module_invoker.hpp"
#include "hipSYCL/runtime/error.hpp"

#include "../module_invoker.hpp"

namespace hipsycl {
namespace rt {

class ze_queue;

class ze_module_invoker : public module_invoker {
public:
  ze_module_invoker(ze_queue* queue)
  : _queue{queue} {}

  virtual result
  submit_kernel(module_id_t id, const std::string &module_variant,
                const std::string *module_image, const rt::range<3> &num_groups,
                const rt::range<3>& group_size, unsigned local_mem_size,
                void **args, std::size_t* arg_sizes, std::size_t num_args,
                const std::string &kernel_name_tag,
                const std::string &kernel_body_name) override;
private:
  ze_queue* _queue;
};

class ze_module {
public:
  ze_module(ze_context_handle_t ctx, ze_device_handle_t dev, module_id_t id,
            const std::string& variant, const std::string *module_image);

  ~ze_module();
  
  ze_module(const ze_module&) = delete;
  ze_module& operator=(const ze_module&) = delete;

  ze_module_handle_t get_handle() const;
  module_id_t get_id() const;
  ze_device_handle_t get_device() const;
  result get_build_status() const;
  const std::string& get_variant() const;

  result obtain_kernel(const std::string& name, ze_kernel_handle_t& out) const;
  result obtain_kernel(const std::string &name,
                       const std::string &fallback_name,
                       ze_kernel_handle_t &out) const;

private:

  result _build_status;
  module_id_t _id;
  std::string _variant;
  ze_device_handle_t _dev;
  ze_module_handle_t _handle;
  mutable std::vector<std::pair<std::string, ze_kernel_handle_t>> _kernels;
};

}
}

#endif
