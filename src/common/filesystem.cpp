/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
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

#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/config.hpp"


#ifndef _WIN32
#include <dlfcn.h>
#else
#include <windows.h> 
#endif

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;


namespace hipsycl {
namespace common {
namespace filesystem {

std::string get_install_directory() {

  std::vector<fs::path> paths;
#ifndef _WIN32
  Dl_info info;
  if (dladdr(reinterpret_cast<void*>(&get_install_directory), &info)) {
    auto lib_path = fs::path{info.dli_fname}.parent_path();
    if(lib_path.has_parent_path())
      paths.emplace_back(lib_path.parent_path());
  }
  
#else
  if(HMODULE handle = GetModuleHandleA(HIPSYCL_COMMON_LIBRARY_NAME))
  {
    std::vector<char> path_buffer(MAX_PATH);
    if(GetModuleFileNameA(handle, path_buffer.data(), path_buffer.size()))
    {
      auto lib_path = fs::path{path_buffer.data()}.parent_path();
      if(lib_path.has_parent_path())
        paths.emplace_back(lib_path.parent_path());
    }
  }
  
#endif
  if(paths.empty() || !fs::is_directory(paths.back()))
    return fs::path{HIPSYCL_INSTALL_PREFIX}.string();
  return paths.back().string();
}

std::string join_path(const std::string& base, const std::string& extra) {
  return (fs::path(base) / extra).string();
}

std::string
join_path(const std::string &base,
          const std::vector<std::string> &additional_components) {
  std::string current = base;
  for(const auto& extra : additional_components) {
    current = join_path(current, extra);
  }
  return current;
}

std::vector<std::string> list_regular_files(const std::string& directory) {
  fs::path p{directory};
  std::vector<std::string> result;
  for(const fs::directory_entry& entry : fs::directory_iterator(p)) {
    if(fs::is_regular_file(entry.status())) {
      result.push_back(entry.path().string());
    }
  }
  return result;
}

std::vector<std::string> list_regular_files(const std::string& directory,
  const std::string& extension) { 
  
  auto all_files = list_regular_files(directory);
  std::vector<std::string> result;
  for(const auto& f : all_files) {
    if(fs::path(f).extension().string() == extension)
      result.push_back(f);
  }
  return result;
}

}
}
}

