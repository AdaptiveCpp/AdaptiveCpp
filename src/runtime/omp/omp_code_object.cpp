/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2021 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <cassert>
#include <cctype>
#include <sstream>
#include <string>

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/omp/omp_code_object.hpp"

#include "hipSYCL/common/config.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/dylib_loader.hpp"
#include "hipSYCL/runtime/error.hpp"

namespace hipsycl {
namespace rt {

namespace {

result make_shared_library_from_blob(void *&module, const std::string &blob,
                                     const std::string &cache_file) {
  // Write binary image to temporary file
  if (!common::filesystem::atomic_write(cache_file, blob)) {
    HIPSYCL_DEBUG_ERROR << "Could not store JIT kernel library in temporary "
                           "kernel cache in file "
                        << cache_file << std::endl;
    return make_error(
        __hipsycl_here(),
        error_info{"omp_sscp_executable_object: could not store JIT kernel "
                   "library in temporary kernel cache"});
  }

  // now load the same so file
  HIPSYCL_DEBUG_INFO << "Load module: " << cache_file << "\n";
  module = detail::load_library(cache_file, "omp_sscp_executable");

  if (!module)
    return make_error(__hipsycl_here(),
                      error_info{"omp_sscp_executable_object: could not load "
                                 "shared kernel library"});

  return make_success();
}

} // namespace

omp_sscp_executable_object::omp_sscp_executable_object(
    const std::string &binary, hcf_object_id hcf_source,
    const std::vector<std::string> &kernel_names,
    const kernel_configuration &config)
    : _hcf{hcf_source}, _id{config.generate_id()}, _module{nullptr},
      _kernel_cache_path(kernel_cache::get_persistent_cache_file(_id) + ".so") {
  _build_result = build(binary, kernel_names);
}

omp_sscp_executable_object::~omp_sscp_executable_object() {
  if (_module)
    detail::close_library(_module, "omp_sscp_executable");
  if(!common::filesystem::remove(_kernel_cache_path)) {
    HIPSYCL_DEBUG_ERROR << "Could not remove kernel cache file: "
                        << _kernel_cache_path << std::endl;
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
  std::vector<std::string> names;
  names.reserve(_kernels.size());
  std::transform(_kernels.begin(), _kernels.end(), std::back_inserter(names),
                 [](const auto &pair) { return pair.first; });
  return names;
}

void *omp_sscp_executable_object::get_module() const { return _module; }

result omp_sscp_executable_object::build(
    const std::string &source, const std::vector<std::string> &kernel_names) {
  if (_module != nullptr)
    return make_success();

  if (auto result = make_shared_library_from_blob(_module, source, _kernel_cache_path);
      !result.is_success())
    return result;

  // find all kernel symbols
  for (const auto &kernel_name : kernel_names) {
    if (auto kernel = (omp_sscp_kernel *)detail::get_symbol_from_library(
            _module, kernel_name, "omp_sscp_exectuable_object")) {
      _kernels.emplace(kernel_name, kernel);
    } else {
      return make_error(__hipsycl_here(),
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

omp_sscp_executable_object::omp_sscp_kernel *omp_sscp_executable_object::get_kernel(const std::string &backend_kernel_name) const {
  auto it = _kernels.find(backend_kernel_name);
  if (it != _kernels.end())
    return it->second;
  return nullptr;
}

} // namespace rt
} // namespace hipsycl
