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
#ifndef HIPSYCL_HW_MODEL_HPP
#define HIPSYCL_HW_MODEL_HPP

#include <memory>
#include "memcpy.hpp"

namespace hipsycl {
namespace rt {

class backend_manager;
class hw_model
{
public:
  hw_model(backend_manager* backends)
  : _memcpy_model{std::make_unique<memcpy_model>(backends)}
  {}

  memcpy_model *get_memcpy_model() const
  {
    return _memcpy_model.get();
  }

private:
  std::unique_ptr<memcpy_model> _memcpy_model;
};

}
}

#endif