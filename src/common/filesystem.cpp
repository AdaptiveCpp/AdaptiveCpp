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
#include "hipSYCL/common/stable_running_hash.hpp"
#include "hipSYCL/common/debug.hpp"

#include "hipSYCL/runtime/settings.hpp"

#include <fstream>
#include <random>
#include <cassert>

#ifndef _WIN32
#include <dlfcn.h>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#include <limits.h>
#else
#include <windows.h> 
#endif

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;


namespace hipsycl {
namespace common {
namespace filesystem {

namespace {


template<class T>
inline T random_number() {
  thread_local std::random_device rd;
  thread_local std::mt19937 gen{rd()};
  thread_local std::uniform_int_distribution<T> distribution{0};

  return distribution(gen);
}

}

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

bool atomic_write(const std::string &filename, const std::string &data) {
  fs::path p{filename};

  std::string temp_file = std::to_string(random_number<std::size_t>())+".tmp";
  fs::path tmp_path = p.parent_path() / temp_file;

  std::ofstream ostr{tmp_path, std::ios::binary|std::ios::out|std::ios::trunc};
  
  if(!ostr.is_open())
    return false;
  
  ostr.write(data.data(), data.size());
  ostr.close();

  fs::rename(tmp_path, p);

  return true;
}

bool remove(const std::string &filename) {
  try {
    return fs::remove(filename);
  } catch (const fs::filesystem_error &err) {}
  return false;
}

tuningdb::tuningdb() {
#ifndef _WIN32

  auto get_home = [](std::string &home_out, std::string &subdirectory) -> bool {
    if(const char* home = std::getenv("XDG_DATA_HOME")) {
      home_out = home;
      subdirectory = "acpp";
      return true;
    }

    if (const char* home = std::getenv("HOME")) {
      home_out = home;
      subdirectory = ".acpp";
      return true;
    }
        
    const char* home = getpwuid(getuid())->pw_dir;
    if(home) {
      home_out = home;
      subdirectory = ".acpp";
      return true;
    }

    return false;
  };

  if(!rt::try_get_environment_variable("appdb_dir", _base_dir)) {
    std::string home, subdirectory;
    if (get_home(home, subdirectory)) {
      _base_dir = (fs::path{home} / subdirectory).string();
    } else {
      _base_dir = (fs::current_path() / ".acpp").string();
    }
  }

  auto get_app_path = []() -> std::string{

    char result[PATH_MAX];
    ssize_t num_read = readlink("/proc/self/exe", result, PATH_MAX);
    if(num_read >= 0 && num_read <= PATH_MAX - 1) {
      result[num_read] = '\0';
      return std::string{result};
    }
    return std::string{};
  };

  std::string app_path = get_app_path();
  std::string app_subdirectory = "global";
  if(!app_path.empty()) {
    std::string app_filename = fs::path{app_path}.filename().string();

    stable_running_hash h;
    h(app_path.data(), app_path.size());
    app_subdirectory = app_filename + "-" + std::to_string(h.get_current_hash());
  }

  _this_app_dir = (fs::path{_base_dir} / "apps" / app_subdirectory).string();
  
#else
  _base_dir = (fs::current_path() / ".acpp").string();
  _this_app_dir = (fs::path{_base_dir} / "apps" / "global").string();
#endif

   _jit_cache_dir = (fs::path{_this_app_dir} / "jit-cache").string();

  fs::create_directories(_base_dir);
  fs::create_directories(_this_app_dir);
  fs::create_directories(_jit_cache_dir);
}

}
}
}

