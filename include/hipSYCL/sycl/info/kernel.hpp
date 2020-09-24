/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_INFO_KERNEL_HPP
#define HIPSYCL_INFO_KERNEL_HPP

#include "../types.hpp"
#include "param_traits.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"

namespace hipsycl {
namespace sycl {

class context;
class program;

namespace info {

enum class kernel : int {
  function_name,
  num_args,
  context,
  program,
  reference_count,
  attributes
};

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::function_name, string_class);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::num_args, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::context, sycl::context);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::program, sycl::program);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::reference_count, detail::u_int);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel, kernel::attributes, string_class);

enum class kernel_work_group : int
{
  global_work_size,
  work_group_size,
  compile_work_group_size,
  preferred_work_group_size_multiple,
  private_mem_size
};

HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel_work_group, 
                                kernel_work_group::global_work_size, sycl::range<3>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel_work_group, 
                                kernel_work_group::work_group_size, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel_work_group, 
                                kernel_work_group::compile_work_group_size, sycl::range<3>);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel_work_group, 
                                kernel_work_group::preferred_work_group_size_multiple, size_t);
HIPSYCL_PARAM_TRAIT_RETURN_VALUE(kernel_work_group, 
                                kernel_work_group::private_mem_size, detail::u_long);

} // info
} // sycl
} // hipsycl


#endif
