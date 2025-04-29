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

