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

#ifndef HIPSYCL_APPLICATION_HPP
#define HIPSYCL_APPLICATION_HPP

#include <memory>

#include "backend.hpp"
#include "device_id.hpp"
#include "settings.hpp"

namespace hipsycl {
namespace rt {

class dag_manager;
class runtime;
class async_error_list;

class application
{
public:
  static settings& get_settings();
  // Should only be invoked from the SYCL interface, not
  // from the runtime or kernel launchers.
  static std::shared_ptr<runtime> get_runtime_pointer();
  static async_error_list& errors();

  application() = delete;
};

class persistent_runtime {
public:
  persistent_runtime() : _rt{nullptr} {
    if(application::get_settings().get<setting::persistent_runtime>()) {
      _rt = application::get_runtime_pointer();
    }
  }
private:
  std::shared_ptr<runtime> _rt;
};

class runtime_keep_alive_token {
public:
  runtime_keep_alive_token();

  runtime* get() const;
private:
  std::shared_ptr<runtime> _rt;
};

}
}


#endif
