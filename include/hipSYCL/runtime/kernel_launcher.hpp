/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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

#ifndef HIPSYCL_KERNEL_LAUNCHER_HPP
#define HIPSYCL_KERNEL_LAUNCHER_HPP

#include <limits>
#include <optional>
#include <vector>
#include <memory>

#include "hipSYCL/common/small_vector.hpp"
#include "hipSYCL/runtime/dag_node.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/runtime/error.hpp"
#include "hipSYCL/runtime/hints.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/glue/kernel_configuration.hpp"

#include "backend.hpp"

namespace hipsycl {
namespace rt {

class multipass_code_object_invoker;
class sscp_code_object_invoker;

enum class kernel_type {
  single_task,
  basic_parallel_for,
  ndrange_parallel_for,
  hierarchical_parallel_for,
  scoped_parallel_for,
  custom
};

class backend_kernel_launch_capabilities {
public:
  void provide_multipass_invoker(multipass_code_object_invoker* invoker) {
    _multipass_invoker = invoker;
  }

  void provide_sscp_invoker(sscp_code_object_invoker* invoker) {
    _sscp_invoker = invoker;
  }

  std::optional<multipass_code_object_invoker*> get_multipass_invoker() const {
    if(_multipass_invoker)
      return _multipass_invoker;
    return {};
  }

  std::optional<sscp_code_object_invoker*> get_sscp_invoker() const {
    if(_sscp_invoker)
      return _sscp_invoker;
    return {};
  }
private:
  multipass_code_object_invoker* _multipass_invoker = nullptr;
  sscp_code_object_invoker* _sscp_invoker = nullptr;
};

class backend_kernel_launcher
{
public:
  virtual ~backend_kernel_launcher(){}

  virtual int get_backend_score(backend_id b) const = 0;
  virtual kernel_type get_kernel_type() const = 0;
  // Additional backend-specific parameters (e.g. queue)
  virtual void set_params(void*) = 0;
  virtual void invoke(dag_node *node,
                      const glue::kernel_configuration &config) = 0;

  void set_backend_capabilities(const backend_kernel_launch_capabilities& cap) {
    _capabilities = cap;
  }

  const backend_kernel_launch_capabilities& get_launch_capabilities() const {
    return _capabilities;
  }
private:
  backend_kernel_launch_capabilities _capabilities;
};

class kernel_launcher
{
public:
  kernel_launcher(
      common::auto_small_vector<std::unique_ptr<backend_kernel_launcher>> kernels)
  : _kernels{std::move(kernels)}
  {}

  kernel_launcher(const kernel_launcher &) = delete;

  void invoke(backend_id id, rt::dag_node_ptr node) const {
    find_launcher(id)->invoke(node.get(), _kernel_config);
  }

  backend_kernel_launcher* find_launcher(backend_id id) const {
    int max_score = -1;
    backend_kernel_launcher* selected_launcher = nullptr;

    for (auto &backend_launcher : _kernels) {
      int score = backend_launcher->get_backend_score(id);
      if (score >= 0 && score > max_score) {
        max_score = score;
        selected_launcher = backend_launcher.get();
      }
    }
    if(!selected_launcher){
      register_error(
          __hipsycl_here(),
          error_info{"No kernel launcher is present for requested backend",
                    error_type::invalid_parameter_error});
    }
    return selected_launcher;
  }

  const glue::kernel_configuration& get_kernel_configuration() const {
    return _kernel_config;
  }
private:
  common::auto_small_vector<std::unique_ptr<backend_kernel_launcher>>
      _kernels;
  glue::kernel_configuration _kernel_config;
};


} // namespace rt
} // namespace hipsycl

#endif
