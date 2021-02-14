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


#include "hipSYCL/runtime/backend_loader.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/config.hpp"

#include <cassert>
#include <filesystem>
#include <dlfcn.h>

namespace {

void close_plugin(void *handle) {
  if (int err = dlclose(handle)) {
    HIPSYCL_DEBUG_ERROR << "backend_loader: dlclose() failed" << std::endl;
  }
}

bool load_plugin(const std::string &filename, void *&handle_out,
                 std::string &backend_name_out) {
  
  if(void *handle = dlopen(filename.c_str(), RTLD_NOW)) {

    void *symbol = dlsym(handle, "hipsycl_backend_plugin_get_name");

    if (char *err = dlerror()) {
      HIPSYCL_DEBUG_ERROR << "backend_loader: Could not retrieve backend name"
                          << err << std::endl;

      close_plugin(handle);
      return false;
    } else {
      auto get_name =
          reinterpret_cast<decltype(&hipsycl_backend_plugin_get_name)>(symbol);

      handle_out = handle;
      backend_name_out = get_name();

      return true;
    }
  } else {
    HIPSYCL_DEBUG_ERROR << "backend_loader: Could not load backend plugin: "
                        << filename << std::endl;
    if (char *err = dlerror()) {
      HIPSYCL_DEBUG_ERROR << err << std::endl;
    }

    return false;
  } 

  
}

hipsycl::rt::backend *create_backend(void *plugin_handle) {
  assert(plugin_handle);

  void *symbol = dlsym(plugin_handle, "hipsycl_backend_plugin_create");
  char *err = dlerror();
  if (err) {
    HIPSYCL_DEBUG_ERROR
        << "backend_loader: Could not find symbol for backend creation" << err
        << std::endl;
    
    return nullptr;
  }
  
  auto create_backend_func =
      reinterpret_cast<decltype(&hipsycl_backend_plugin_create)>(symbol);

  return create_backend_func();
}

}

namespace hipsycl {
namespace rt {

void backend_loader::query_backends() {
  std::string install_prefix = HIPSYCL_INSTALL_PREFIX;

  std::filesystem::path backend_lib_path =
      std::filesystem::path{install_prefix} / "lib/hipSYCL";

  std::string shared_lib_extension = ".so";
  
  for (const std::filesystem::directory_entry &entry :
       std::filesystem::directory_iterator(backend_lib_path)) {

    if (entry.is_regular_file()) {
      auto p = entry.path();
      if (p.extension().string() == shared_lib_extension) {
        std::string backend_name;
        void *handle;
        if (load_plugin(p.string(), handle, backend_name)) {
          HIPSYCL_DEBUG_INFO << "Successfully opened plugin: " << p
                             << " for backend '" << backend_name << "'"
                             << std::endl;
          _handles.emplace_back(std::make_pair(backend_name, handle));
        }
      }
    }
    
  }
}

backend_loader::~backend_loader() {
  for (auto &handle : _handles) {
    assert(handle.second);
    close_plugin(handle.second);
  }
}

std::size_t backend_loader::get_num_backends() const { return _handles.size(); }

std::string backend_loader::get_backend_name(std::size_t index) const {
  assert(index < _handles.size());
  return _handles[index].first;
}

bool backend_loader::has_backend(const std::string &name) const {
  for (const auto &h : _handles) {
    if (h.first == name)
      return true;
  }

  return false;
}

backend *backend_loader::create(std::size_t index) const {
  assert(index < _handles.size());
  
  return create_backend(_handles[index].second);
}

backend *backend_loader::create(const std::string &name) const {
  
  for (std::size_t i = 0; i < _handles.size(); ++i) {
    if (_handles[i].first == name)
      return create(i);
  }

  return nullptr;
}

}
} // namespace hipsycl
