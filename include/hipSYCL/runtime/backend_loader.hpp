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

#ifndef HIPSYCL_BACKEND_LOADER_HPP
#define HIPSYCL_BACKEND_LOADER_HPP


#include <string>
#include <vector>
#include <utility>

namespace hipsycl::rt {
class backend;
}

#ifndef _WIN32
#define HIPSYCL_PLUGIN_API_EXPORT extern "C"
#else
#define HIPSYCL_PLUGIN_API_EXPORT extern "C" __declspec(dllexport)
#endif

HIPSYCL_PLUGIN_API_EXPORT
hipsycl::rt::backend *hipsycl_backend_plugin_create();

HIPSYCL_PLUGIN_API_EXPORT
const char* hipsycl_backend_plugin_get_name();


namespace hipsycl {
namespace rt {

class backend_loader {
public:
  ~backend_loader();

  void query_backends();
  
  std::size_t get_num_backends() const;
  std::string get_backend_name(std::size_t index) const;
  bool has_backend(const std::string &name) const;

  backend *create(std::size_t index) const;
  backend *create(const std::string &name) const;

private:
  using handle_t = void*;
  std::vector<std::pair<std::string, handle_t>> _handles;
};

}
}

#endif

