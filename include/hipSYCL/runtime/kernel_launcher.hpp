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
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/kernel_type.hpp"
#include "hipSYCL/glue/kernel_launcher_data.hpp"

#include "backend.hpp"

namespace hipsycl {
namespace rt {

class multipass_code_object_invoker;
class sscp_code_object_invoker;


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
                      const kernel_configuration &config) = 0;

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
      const glue::kernel_launcher_data& static_data,
      common::auto_small_vector<std::unique_ptr<backend_kernel_launcher>> kernels)
  : _static_data{static_data}, _kernels{std::move(kernels)}
  {}

  rt::result invoke(backend_id id, void *params,
                    const rt::backend_kernel_launch_capabilities& cap,
                    rt::dag_node* node) const {


    for(auto& backend_launcher : _kernels) {
      // Just pick first accepting launcher for now - in practice there can
      // be no conflict anyway since SSCP is handled separately
      if(backend_launcher->get_backend_score(id) >= 0) {
        backend_launcher->set_params(params);
        backend_launcher->set_backend_capabilities(cap);
        backend_launcher->invoke(node, _kernel_config);
        return make_success();
      }
    }

    if (cap.get_sscp_invoker().has_value() &&
        (_static_data.sscp_kernel_id || _static_data.custom_op)) {
      return _static_data.sscp_invoker(_static_data, node, _kernel_config,
                                       cap, params);
    }

    return make_error(
          __acpp_here(),
          error_info{"No kernel launcher is present for requested backend",
                    error_type::invalid_parameter_error});
  }

  const kernel_configuration& get_kernel_configuration() const {
    return _kernel_config;
  }
private:
  
  common::auto_small_vector<std::unique_ptr<backend_kernel_launcher>>
      _kernels;
  glue::kernel_launcher_data _static_data;
  kernel_configuration _kernel_config;
};


} // namespace rt
} // namespace hipsycl

#endif
