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

#include <vector>
#include <memory>

#include "hipSYCL/sycl/id.hpp"
#include "hipSYCL/sycl/range.hpp"



#include "backend.hpp"

namespace hipsycl {
namespace rt {

enum class kernel_type {
  single_task,
  basic_parallel_for,
  ndrange_parallel_for,
  hierarchical_parallel_for
};

class backend_kernel_launcher
{
public:
  virtual ~backend_kernel_launcher(){}

  virtual backend_id get_backend() const = 0;
  virtual kernel_type get_kernel_type() const = 0;
  virtual void invoke() = 0;
};

class kernel_launcher
{
public:
  kernel_launcher(
      std::vector<std::unique_ptr<backend_kernel_launcher>>&& kernels);

  void invoke(backend_id id) {
    /*for (auto &backend_launcher : _kernels) {
      if (backend_launcher->get_backend() == id) {
        backend_launcher->invoke();
        return;
      }
    }*/

    assert(false && "Could not find kernel launcher for the specified backend");
  }

private:
  //std::vector<std::unique_ptr<backend_kernel_launcher>> _kernels;
};


} // namespace rt
} // namespace hipsycl

#endif
