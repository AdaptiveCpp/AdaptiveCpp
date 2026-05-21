/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause
#include <algorithm>
#include <cassert>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>

#ifndef _WIN32
#include <unistd.h>
#else
#include <windows.h>
#endif

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/omp/omp_code_object.hpp"

#include "hipSYCL/common/config.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/dylib_loader.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/settings.hpp"

namespace hipsycl {
namespace rt {

namespace {

std::string make_unique_temp_cache_file_path(kernel_configuration::id_type id) {
  std::string persistent_cache_file =
      kernel_cache::get_persistent_cache_file(id);

  const auto now =
      std::chrono::high_resolution_clock::now().time_since_epoch().count();
  const std::size_t tid_hash =
      std::hash<std::thread::id>{}(std::this_thread::get_id());

#ifndef _WIN32
  const auto pid = static_cast<std::uint64_t>(getpid());
#else
  const auto pid = static_cast<std::uint64_t>(GetCurrentProcessId());
#endif

  static std::mutex rng_mutex;
  static std::mt19937_64 rng{static_cast<std::uint64_t>(std::random_device{}())};
  static std::uniform_int_distribution<std::uint64_t> dist;

  std::uint64_t random = 0;
  {
    std::lock_guard<std::mutex> lock{rng_mutex};
    random = dist(rng);
  }

  return persistent_cache_file + ".tmp." + std::to_string(now) + "." +
         std::to_string(pid) + "." + std::to_string(tid_hash) + "." +
         std::to_string(random) + "." + ACPP_SHARED_LIBRARY_EXTENSION;
}

result make_shared_library_from_blob(void *&module, const std::string &blob,
                                     const std::string &cache_file) {
  // Write binary image to temporary file
  if (!common::filesystem::atomic_write(cache_file, blob)) {
    HIPSYCL_DEBUG_ERROR << "Could not store JIT kernel library in temporary "
                           "kernel cache in file "
                        << cache_file << std::endl;
    return make_error(
        __acpp_here(),
        error_info{"omp_sscp_executable_object: could not store JIT kernel "
                   "library in temporary kernel cache"});
  }

  // now load the same so file
  HIPSYCL_DEBUG_INFO << "Load module: " << cache_file << "\n";
  std::string message;
  module = common::load_library(cache_file, message);
  if (!message.empty()) {
    HIPSYCL_DEBUG_WARNING << "[omp_sscp_executable_object] " << message << std::endl;
  }

  if (!module)
    return make_error(__acpp_here(),
                      error_info{"omp_sscp_executable_object: could not load "
                                 "shared kernel library"});

  return make_success();
}

} // namespace

omp_sscp_executable_object::omp_sscp_executable_object(
    const std::string &binary, hcf_object_id hcf_source,
    const std::vector<std::string> &kernel_names,
    const kernel_configuration &config)
    : _hcf{hcf_source}, _id{config.generate_id()}, _kernel_cache_path{},
      _build_result{}, _module{nullptr},
      _remove_kernel_cache_file_in_dtor{false} {

  if (application::get_settings().get<setting::no_jit_cache_population>()) {
    // In this mode, JIT output should not populate persistent cache entries.
    // Use a unique temporary file to avoid cross-process collisions.
    _kernel_cache_path = make_unique_temp_cache_file_path(_id);
    _remove_kernel_cache_file_in_dtor = true;
  } else {
    // Use the persistent cache path and do not remove it in the destructor,
    // because other processes may concurrently use this file.
    _kernel_cache_path =
        kernel_cache::get_persistent_cache_file(_id) + "." + ACPP_SHARED_LIBRARY_EXTENSION;
  }

  _build_result = build(binary, kernel_names);
}

omp_sscp_executable_object::~omp_sscp_executable_object() {
  if (_module){
    std::string message;
    common::close_library(_module, message);
    if (!message.empty()) {
      HIPSYCL_DEBUG_ERROR << "[omp_sscp_executable_object] " << message << std::endl;
    }
  }
  if (_remove_kernel_cache_file_in_dtor) {
    if(!common::filesystem::remove(_kernel_cache_path)) {
      HIPSYCL_DEBUG_ERROR << "Could not remove kernel cache file: "
                          << _kernel_cache_path << std::endl;
    }
  }
}

result omp_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state omp_sscp_executable_object::state() const {
  return _module ? code_object_state::executable : code_object_state::invalid;
}

code_format omp_sscp_executable_object::format() const {
  return code_format::native_isa;
}

backend_id omp_sscp_executable_object::managing_backend() const {
  return backend_id::omp;
}

hcf_object_id omp_sscp_executable_object::hcf_source() const { return _hcf; }

std::string omp_sscp_executable_object::target_arch() const {
  return "native-host";
}

compilation_flow omp_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
omp_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

void *omp_sscp_executable_object::get_module() const { return _module; }

result omp_sscp_executable_object::build(
    const std::string &source, const std::vector<std::string> &kernel_names) {
    
  if (_module != nullptr)
    return make_success();

  if (auto result = make_shared_library_from_blob(_module, source, _kernel_cache_path);
      !result.is_success())
    return result;

  _kernel_names = kernel_names;
  // find all kernel symbols
  for (const auto &kernel_name : _kernel_names) {
    std::string message;
    if (auto kernel = (omp_sscp_kernel *)common::get_symbol_from_library(
            _module, kernel_name, message)) {
      _kernels.emplace(kernel_name, kernel);
    } else {
      if (!message.empty()) {
        HIPSYCL_DEBUG_WARNING << "[omp_sscp_executable_object] " << message << std::endl;
      }
      return make_error(__acpp_here(),
                        error_info{"omp_sscp_executable_object: could not load "
                                   "kernel from shared library"});
    }
  }
  return make_success();
}

bool omp_sscp_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for (const auto &[kernel_name, kernel] : _kernels) {
    if (kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

omp_sscp_executable_object::omp_sscp_kernel *
omp_sscp_executable_object::get_kernel(std::string_view backend_kernel_name) const {
  auto it = _kernels.find(backend_kernel_name);
  if (it != _kernels.end())
    return it->second;
  return nullptr;
}

} // namespace rt
} // namespace hipsycl
