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

#include "hipSYCL/sycl/libkernel/range.hpp"
#include "info.hpp"
#include "../types.hpp"

namespace hipsycl {
namespace sycl {

class context;
class program;

namespace info {

namespace kernel {
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(function_name, string_class);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(num_args, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(context, sycl::context);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(program, sycl::program);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(reference_count, detail::u_int);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(attributes, string_class);
};

namespace kernel_device_specific {

struct global_work_size {
  using return_type = range<3>;
};
struct work_group_size {
  using return_type = std::size_t;
};
struct compile_work_group_size {
  using return_type = range<3>;
};
struct preferred_work_group_size_multiple {
  using return_type = std::size_t;
};
struct private_mem_size {
  using return_type = std::size_t;
};
struct max_num_sub_groups {
  using return_type = uint32_t;
};
struct compile_num_sub_groups {
  using return_type = uint32_t;
};
struct max_sub_group_size {
  using return_type = uint32_t;
};
struct compile_sub_group_size {
  using return_type = uint32_t;
};

}  // namespace kernel_device_specific

namespace kernel_work_group
{
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_work_size, sycl::range<3>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(work_group_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(compile_work_group_size, sycl::range<3>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_work_group_size_multiple, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(private_mem_size, detail::u_long);
};

} // info
} // sycl
} // hipsycl

#endif
