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
#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h> 
#endif

namespace {

void close_plugin(void *handle) {
#ifndef _WIN32
  if (int err = dlclose(handle)) {
    HIPSYCL_DEBUG_ERROR << "backend_loader: dlclose() failed" << std::endl;
  }
#else
  if (!FreeLibrary(static_cast<HMODULE>(handle))) {
    HIPSYCL_DEBUG_ERROR << "backend_loader: FreeLibrary() failed" << std::endl;
  }
#endif
}

void* load_library(const std::string &filename)
{
#ifndef _WIN32
  if(void *handle = dlopen(filename.c_str(), RTLD_NOW)) {
    return handle;
  } else {
    HIPSYCL_DEBUG_ERROR << "backend_loader: Could not load backend plugin: "
                        << filename << std::endl;
    if (char *err = dlerror()) {
      HIPSYCL_DEBUG_ERROR << err << std::endl;
    }
  }
#else
  if(HMODULE handle = LoadLibraryA(filename.c_str())) {
    return static_cast<void*>(handle);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_ERROR << "backend_loader: Could not load backend plugin: "
                        << filename << " with: " << GetLastError() << std::endl;
  }
#endif
  return nullptr;
}

void* get_symbol_from_library(void* handle, const std::string& symbolName)
{
#ifndef _WIN32
  void *symbol = dlsym(handle, symbolName.c_str());
  if(char *err = dlerror()) {
    HIPSYCL_DEBUG_ERROR << "backend_loader: Could not find symbol name: "
                        << symbolName << std::endl;
    HIPSYCL_DEBUG_ERROR << err << std::endl;
  } else {
    return symbol;
  }
#else
  if(FARPROC symbol = GetProcAddress(static_cast<HMODULE>(handle), symbolName.c_str())) {
    return reinterpret_cast<void*>(symbol);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_ERROR << "backend_loader: Could not find symbol name: "
                        << symbolName << " with: " << GetLastError() << std::endl;
  }
#endif
  return nullptr;
}

bool load_plugin(const std::string &filename, void *&handle_out,
                 std::string &backend_name_out) {
  if(void *handle = load_library(filename)) {
    if(void* symbol = get_symbol_from_library(handle, "hipsycl_backend_plugin_get_name"))
    {
      auto get_name =
          reinterpret_cast<decltype(&hipsycl_backend_plugin_get_name)>(symbol);

      handle_out = handle;
      backend_name_out = get_name();

      return true;
    } else {
      close_plugin(handle);
      return false;
    }
  } else {
    return false;
  } 
}

hipsycl::rt::backend *create_backend(void *plugin_handle) {
  assert(plugin_handle);

  if(void *symbol = get_symbol_from_library(plugin_handle, "hipsycl_backend_plugin_create"))
  {
    auto create_backend_func =
        reinterpret_cast<decltype(&hipsycl_backend_plugin_create)>(symbol);

    return create_backend_func();
  }
  return nullptr;
}

}

namespace hipsycl {
namespace rt {

void backend_loader::query_backends() {
  std::string install_prefix = HIPSYCL_INSTALL_PREFIX;

  std::filesystem::path backend_lib_path =
      std::filesystem::path{install_prefix} / "lib/hipSYCL";

#ifndef _WIN32
  std::string shared_lib_extension = ".so";
#else
  std::string shared_lib_extension = ".dll";
#endif
  
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
