/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/async_errors.hpp"
#include "hipSYCL/runtime/dag_manager.hpp"
#include "hipSYCL/runtime/runtime.hpp"
#include "hipSYCL/runtime/hw_model/hw_model.hpp"
#include "hipSYCL/runtime/settings.hpp"
#include <memory>
#include <mutex>
#include <atomic>

namespace hipsycl {
namespace rt {

runtime_keep_alive_token::runtime_keep_alive_token()
: _rt{application::get_runtime_pointer()} {
  assert(_rt);
}

runtime* runtime_keep_alive_token::get() const {
  return _rt.get();
}

std::shared_ptr<runtime> application::get_runtime_pointer() {
  static std::mutex mutex;
  static std::weak_ptr<runtime> rt;

  std::lock_guard<std::mutex> lock{mutex};

  std::shared_ptr<runtime> rt_ptr = rt.lock();
  if(!rt_ptr) {
    rt_ptr = std::make_shared<runtime>();
    rt = rt_ptr;
  }
  assert(rt_ptr);
  return rt_ptr;
}

settings &application::get_settings() {
  static settings s;
  return s;
}

async_error_list& application::errors() {
  static async_error_list errors;
  return errors;
}


}
}

