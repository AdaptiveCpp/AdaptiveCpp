/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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


#ifndef HIPSYCL_RUNTIME_HPP
#define HIPSYCL_RUNTIME_HPP

#include "dag_manager.hpp"
#include "backend.hpp"
#include "async_errors.hpp"
#include "hipSYCL/common/debug.hpp"

#include <iostream>
namespace hipsycl {
namespace rt {


class runtime
{
public:

  runtime()
  {
    HIPSYCL_DEBUG_INFO << "runtime: ******* rt2 launch initiated ********"
                       << std::endl;
  }

  ~runtime()
  {
    HIPSYCL_DEBUG_INFO << "runtime: ******* rt shutdown ********"
                       << std::endl;
  }

  dag_manager& dag()
  { return _dag_manager; }

  const dag_manager& dag() const
  { return _dag_manager; }

  backend_manager &backends() { return _backends; }

  const backend_manager &backends() const { return _backends; }

  async_error_list& errors() { return _errors; }
  const async_error_list& errors() const { return _errors; }

private:

  dag_manager _dag_manager;
  backend_manager _backends;
  async_error_list _errors;
};

}
}


#endif
