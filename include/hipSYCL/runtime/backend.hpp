/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_RUNTIME_BACKEND_HPP
#define HIPSYCL_RUNTIME_BACKEND_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "device_id.hpp"
#include "backend_loader.hpp"

namespace hipsycl {
namespace rt {

class backend_executor;
class backend_allocator;
class backend_hardware_manager;
class hw_model;

class backend
{
public:
  virtual api_platform get_api_platform() const = 0;
  virtual hardware_platform get_hardware_platform() const = 0;
  virtual backend_id get_unique_backend_id() const = 0;

  virtual backend_hardware_manager* get_hardware_manager() const = 0;
  virtual backend_executor* get_executor(device_id dev) const = 0;
  virtual backend_allocator *get_allocator(device_id dev) const = 0;

  virtual std::string get_name() const = 0;

  virtual ~backend() {}

  backend_descriptor get_backend_descriptor() const {
    return backend_descriptor{this->get_hardware_platform(),
                              this->get_api_platform()};
  }
};

class backend_manager
{
public:
  using backend_list_type =
      std::vector<std::unique_ptr<backend>>;

  backend_manager();
  ~backend_manager();
  
  backend* get(backend_id) const;
  hw_model& hardware_model();
  const hw_model& hardware_model() const;

  template<class F>
  void for_each_backend(F f)
  {
    for(auto& b : _backends){
      f(b.get());
    }
  }

private:
  backend_loader _loader;
  backend_list_type _backends;

  std::unique_ptr<hw_model> _hw_model;
};

}
}

#endif
