/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2022 Aksel Alpay and contributors
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

#include "hipSYCL/runtime/kernel_cache.hpp"

namespace hipsycl {
namespace rt {

kernel_cache& kernel_cache::get() {
  static kernel_cache c;
  return c;
}

void kernel_cache::register_hcf_object(const common::hcf_container &obj) {

  if (!obj.root_node()->has_key("object-id")) {
    HIPSYCL_DEBUG_ERROR
        << "kernel_cache: Invalid hcf object (missing object id)" << std::endl;
  }
  const std::string *data = obj.root_node()->get_value("object-id");

  assert(data);
  hcf_object_id id = std::stoull(*data);
  HIPSYCL_DEBUG_INFO << "kernel_cache: Registering HCF object " << id << "..." << std::endl;

  if (_hcf_objects.count(id) > 0) {
    HIPSYCL_DEBUG_ERROR
        << "kernel_cache: Detected hcf object id collision " << id
        << ", this should not happen. Some kernels might be unavailable."
        << std::endl;
  } else
    _hcf_objects[id] = obj;
}

const kernel_cache::kernel_name_index_t*
kernel_cache::get_global_kernel_index(const std::string &kernel_name) const {
  auto it = _kernel_index_map.find(kernel_name);
  if(it == _kernel_index_map.end())
    return nullptr;
  return &(it->second);
}

const common::hcf_container* kernel_cache::get_hcf(hcf_object_id obj) const {
  auto it = _hcf_objects.find(obj);
  if(it == _hcf_objects.end())
    return nullptr;
  return &(it->second);
}

void kernel_cache::unload() {
  std::lock_guard<std::mutex> lock{_mutex};

  _kernel_code_objects.clear();
  _code_objects.clear();
}

} // rt
} // hipsycl