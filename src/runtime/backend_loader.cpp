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
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/config.hpp"
#include "hipSYCL/runtime/device_id.hpp"

#include <cassert>

#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h> 
#endif

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;

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
    HIPSYCL_DEBUG_WARNING << "backend_loader: Could not load backend plugin: "
                        << filename << std::endl;
    if (char *err = dlerror()) {
      HIPSYCL_DEBUG_WARNING << err << std::endl;
    }
  }
#else
  if(HMODULE handle = LoadLibraryA(filename.c_str())) {
    return static_cast<void*>(handle);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at
    // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_WARNING << "backend_loader: Could not load backend plugin: "
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
    HIPSYCL_DEBUG_WARNING << "backend_loader: Could not find symbol name: "
                        << symbolName << std::endl;
    HIPSYCL_DEBUG_WARNING << err << std::endl;
  } else {
    return symbol;
  }
#else
  if(FARPROC symbol = GetProcAddress(static_cast<HMODULE>(handle), symbolName.c_str())) {
    return reinterpret_cast<void*>(symbol);
  } else {
    // too lazy to use FormatMessage bs right now, so look up the error at
    // https://docs.microsoft.com/en-us/windows/win32/debug/system-error-codes
    HIPSYCL_DEBUG_WARNING << "backend_loader: Could not find symbol name: "
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

std::vector<fs::path> get_plugin_search_paths()
{
  std::vector<fs::path> paths;
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_plugin_search_paths), &info)) {
    paths.emplace_back(fs::path{info.dli_fname}.parent_path() / "hipSYCL");
  }
  const auto install_prefixed_path = fs::path{HIPSYCL_INSTALL_PREFIX} / "lib" / "hipSYCL";
#else
  if(HMODULE handle = GetModuleHandleA(HIPSYCL_RT_LIBRARY_NAME))
  {
    std::vector<char> path_buffer(MAX_PATH);
    if(GetModuleFileNameA(handle, path_buffer.data(), path_buffer.size()))
    {
      paths.emplace_back(std::filesystem::path{path_buffer.data()}.parent_path() / "hipSYCL");
    }
  }
  const auto install_prefixed_path = std::filesystem::path{HIPSYCL_INSTALL_PREFIX} / "bin" / "hipSYCL";
#endif

  if(paths.empty()
      || !fs::is_directory(paths.back())
      || (fs::is_directory(install_prefixed_path)
          && !fs::equivalent(install_prefixed_path, paths.back())))
    paths.emplace_back(std::move(install_prefixed_path));
  return paths;
}

bool is_plugin_active(const std::string& name)
{
  auto backends_active = hipsycl::rt::application::get_settings().get<hipsycl::rt::setting::visibility_mask>();
  if(backends_active.empty())
    return true;
  if(name == "omp") // we always need a cpu backend
    return true;

  hipsycl::rt::backend_id id;
  if(name == "cuda") {
    id = hipsycl::rt::backend_id::cuda;
  } else if(name == "hip") {
    id = hipsycl::rt::backend_id::hip;
  } else if(name == "ze") {
    id = hipsycl::rt::backend_id::level_zero;
  } else if(name == "ocl") {
    id = hipsycl::rt::backend_id::ocl;
  }
  return backends_active.find(id) != backends_active.cend();
}

}

namespace hipsycl {
namespace rt {

void backend_loader::query_backends() {
  std::vector<fs::path> backend_lib_paths = get_plugin_search_paths();

#ifdef __APPLE__
  std::string shared_lib_extension = ".dylib";
#elif defined(_WIN32)
  std::string shared_lib_extension = ".dll";
#else
  std::string shared_lib_extension = ".so";
#endif

  for(const fs::path& backend_lib_path : backend_lib_paths) {
    if(!fs::is_directory(backend_lib_path)) {
      HIPSYCL_DEBUG_INFO << "backend_loader: Backend lib search path candidate does not exists: "
                        << backend_lib_path << std::endl;
      continue;
    }

    HIPSYCL_DEBUG_INFO << "backend_loader: Searching path for backend libs: '"
                      << backend_lib_path << "'" << std::endl;

    for (const fs::directory_entry &entry :
        fs::directory_iterator(backend_lib_path)) {

      if(fs::is_regular_file(entry.status())){
        auto p = entry.path();
        if (p.extension().string() == shared_lib_extension) {
          std::string backend_name;
          void *handle;
          if (load_plugin(p.string(), handle, backend_name)) {
            if(!has_backend(backend_name) && is_plugin_active(backend_name)){
              HIPSYCL_DEBUG_INFO << "backend_loader: Successfully opened plugin: " << p
                                << " for backend '" << backend_name << "'"
                                << std::endl;
              _handles.emplace_back(std::make_pair(backend_name, handle));
            } else {
              close_plugin(handle);
            }
          }
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
