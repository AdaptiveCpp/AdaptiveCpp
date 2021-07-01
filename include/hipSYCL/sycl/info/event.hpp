/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef HIPSYCL_INFO_EVENT_HPP
#define HIPSYCL_INFO_EVENT_HPP

#include <cstdint>

#include "../types.hpp"
#include "param_traits.hpp"

namespace hipsycl {
namespace sycl {
namespace info {

enum class event: int
{
  command_execution_status,
  reference_count
};

enum class event_command_status : int
{
  submitted,
  running,
  complete
};

enum class event_profiling : int
{
  command_submit,
  command_start,
  command_end
};


HIPSYCL_PARAM_TRAIT_RETURN_VALUE(event, event::command_execution_status, event_command_status);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(event, event::reference_count, detail::u_int);

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(event_profiling, event_profiling::command_submit, uint64_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(event_profiling, event_profiling::command_start, uint64_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(event_profiling, event_profiling::command_end, uint64_t);

}
}
}


#endif
