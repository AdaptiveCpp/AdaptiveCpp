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
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(global_work_size, sycl::range<3>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(work_group_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(compile_work_group_size, sycl::range<3>);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(preferred_work_group_size_multiple, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(private_mem_size, size_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_num_sub_groups, uint32_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(compile_num_sub_groups, uint32_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(max_sub_group_size, uint32_t);
  HIPSYCL_DEFINE_INFO_DESCRIPTOR(compile_sub_group_size, uint32_t);
}

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
