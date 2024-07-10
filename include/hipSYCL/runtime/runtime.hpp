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
#ifndef HIPSYCL_RUNTIME_HPP
#define HIPSYCL_RUNTIME_HPP

#include "dag_manager.hpp"
#include "backend.hpp"
#include "settings.hpp"

#include <memory>
#include <iostream>

namespace hipsycl {
namespace rt {

class runtime
{
public:

  runtime();

  ~runtime();

  dag_manager& dag()
  { return _dag_manager; }

  const dag_manager& dag() const
  { return _dag_manager; }

  backend_manager &backends() { return _backends; }

  const backend_manager &backends() const { return _backends; }

private:
  // !! Attention: order is important, as backends have to be still present,
  // when the dag_manager is destructed!
  backend_manager _backends;
  dag_manager _dag_manager;
};



}
}


#endif
