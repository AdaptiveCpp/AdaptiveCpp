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
#ifndef HIPSYCL_PERSISTENT_RUNTIME_HPP
#define HIPSYCL_PERSISTENT_RUNTIME_HPP

#include <memory>
#include "hipSYCL/runtime/application.hpp"

namespace hipsycl {
namespace glue {

class persistent_runtime {
public:
  persistent_runtime() : _rt{nullptr} {
    if(rt::application::get_settings().get<rt::setting::persistent_runtime>()) {
      _rt = rt::application::get_runtime_pointer();
    }
  }
private:
  std::shared_ptr<rt::runtime> _rt;
};

static persistent_runtime persistent_runtime_object;

}
}

#endif
