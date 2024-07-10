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
#ifndef HIPSYCL_RUNTIME_BACKEND_HPP
#define HIPSYCL_RUNTIME_BACKEND_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "device_id.hpp"
#include "backend_loader.hpp"

namespace hipsycl {
namespace rt {

class backend_executor;
class backend_allocator;
class backend_hardware_manager;
class hw_model;
class kernel_cache;

class backend
{
public:
  virtual api_platform get_api_platform() const = 0;
  virtual hardware_platform get_hardware_platform() const = 0;
  virtual backend_id get_unique_backend_id() const = 0;

  virtual backend_hardware_manager* get_hardware_manager() const = 0;
  virtual backend_executor* get_executor(device_id dev) const = 0;
  virtual backend_allocator *get_allocator(device_id dev) const = 0;

  virtual std::string get_name() const = 0;

  virtual ~backend() {}

  backend_descriptor get_backend_descriptor() const {
    return backend_descriptor{this->get_hardware_platform(),
                              this->get_api_platform()};
  }

  // This is optional; backends can use it to expose inorder executors
  // that might be used by in-order queues for explicit scheduling control
  // by the user.
  //
  // If unsupported by the backend, returns nullptr.
  //
  // priority can be used to define an inorder executor with particular execution
  // priority. It is backend-specific if or how this will affect execution.
  virtual std::unique_ptr<backend_executor>
  create_inorder_executor(device_id dev, int priority) = 0;
};

class backend_manager
{
public:
  using backend_list_type =
      std::vector<std::unique_ptr<backend>>;

  backend_manager();
  ~backend_manager();
  
  backend* get(backend_id) const;
  hw_model& hardware_model();
  const hw_model& hardware_model() const;

  template<class F>
  void for_each_backend(F f)
  {
    for(auto& b : _backends){
      f(b.get());
    }
  }

private:
  backend_loader _loader;
  backend_list_type _backends;

  std::unique_ptr<hw_model> _hw_model;
  std::shared_ptr<kernel_cache> _kernel_cache;
};

}
}

#endif
