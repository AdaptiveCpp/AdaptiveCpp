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

#include <unistd.h>

#include "hipSYCL/runtime/backend.hpp"
#include "hipSYCL/runtime/omp/omp_code_object.hpp"

#include "hipSYCL/common/config.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/common/filesystem.hpp"
#include "hipSYCL/common/hcf_container.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"
#include "hipSYCL/runtime/device_id.hpp"
#include "hipSYCL/runtime/dylib_loader.hpp"
#include "hipSYCL/runtime/error.hpp"

#include HIPSYCL_CXX_FILESYSTEM_HEADER
namespace fs = HIPSYCL_CXX_FILESYSTEM_NAMESPACE;

namespace hipsycl {
namespace rt {

namespace {

result make_shared_library_from_blob(void *&module,
                                     const std::string &cache_file) {

  // assume we have source == so file path
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
    const glue::kernel_configuration &config)
    : _hcf{hcf_source}, _kernel_names{kernel_names}, _id{config.generate_id()},
      _module{nullptr},
      _kernel_cache_path(kernel_cache::get_persistent_cache_file(_id) + "-" +
                         std::to_string(getpid()) + ".so") {
  _build_result = build(binary);
}

omp_sscp_executable_object::~omp_sscp_executable_object() {
  detail::close_library(_module, "omp_sscp_executable");
  try {
    if (!fs::remove(_kernel_cache_path)) {
      HIPSYCL_DEBUG_INFO << "omp_sscp_executable: could not cleanup "
                         << _kernel_cache_path << ".\n";
    }
  } catch (const fs::filesystem_error &err) {
    HIPSYCL_DEBUG_ERROR
        << "omp_sscp_executable: filesystem error while cleaning up "
        << _kernel_cache_path << ": " << err.what() << ".\n";
  }
}

result omp_sscp_executable_object::get_build_result() const {
  return _build_result;
}

code_object_state omp_sscp_executable_object::state() const {
  return _module ? code_object_state::executable : code_object_state::invalid;
}

code_format omp_sscp_executable_object::format() const {
  return code_format::ptx;
}

backend_id omp_sscp_executable_object::managing_backend() const {
  return backend_id::omp;
}

hcf_object_id omp_sscp_executable_object::hcf_source() const { return _hcf; }

std::string omp_sscp_executable_object::target_arch() const {
  return "cpu"; // fixme..?
}

compilation_flow omp_sscp_executable_object::source_compilation_flow() const {
  return compilation_flow::sscp;
}

std::vector<std::string>
omp_sscp_executable_object::supported_backend_kernel_names() const {
  return _kernel_names;
}

void *omp_sscp_executable_object::get_module() const { return _module; }

result omp_sscp_executable_object::build(const std::string &source) {
  // Write binary image to temporary file
  if (!common::filesystem::atomic_write(_kernel_cache_path, source)) {
    HIPSYCL_DEBUG_ERROR << "Could not store JIT kernel library in temporary "
                           "kernel cache in file "
                        << _kernel_cache_path << std::endl;
  }

  if (_module != nullptr)
    return make_success();

  return make_shared_library_from_blob(_module, _kernel_cache_path);
}

bool omp_sscp_executable_object::contains(
    const std::string &backend_kernel_name) const {
  for (const auto &kernel_name : _kernel_names) {
    if (kernel_name == backend_kernel_name)
      return true;
  }
  return false;
}

} // namespace rt
} // namespace hipsycl
