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
#ifndef HIPSYCL_KERNEL_HPP
#define HIPSYCL_KERNEL_HPP

#include "context.hpp"
#include "exception.hpp"
#include "program.hpp"
#include "device.hpp"
#include "info/info.hpp"
#include "info/device.hpp"
#include "info/kernel.hpp"
#include <algorithm>
#include <type_traits>

namespace hipsycl {
namespace sycl {

// This class is mostly a dummy implementation
class kernel 
{
private:
  friend class program;
  // The default object is not valid because there is no
  // program or cl_kernel associated with it
  kernel();

  context _ctx;
public:
  template<class Kernel_type>  
  kernel(Kernel_type clKernel, const context& syclContext)
  : _ctx{syclContext}
  {}
  
  /* -- common interface members -- */
  
  //cl_kernel get() const;

  bool is_host() const
  {
    return _ctx.is_host();
  }

  context get_context() const
  {
    return _ctx;
  }

  program get_program() const;
  
  template <typename Param>
  typename Param::return_type get_info() const;

  template <class KernelDeviceSpecificT>
  typename KernelDeviceSpecificT::return_type
  get_info(const device& dev) const {
    using namespace info::kernel_device_specific;

    if constexpr (std::is_same_v<KernelDeviceSpecificT, global_work_size>) {
      throw exception{make_error_code(errc::invalid),
                      "Cannot query global_work_size for this kernel"};
    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        work_group_size>) {
      return dev.get_info<info::device::max_work_group_size>();

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        compile_work_group_size>) {
      return range<3>{0, 0, 0};
    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        preferred_work_group_size_multiple>) {
      return dev.get_info<info::device::sub_group_sizes>()[0];

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        private_mem_size>) {
      return size_t{0};

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        max_num_sub_groups>) {
                                          
      auto subgroups = dev.get_info<info::device::sub_group_sizes>();
      return static_cast<uint32_t>(
          dev.get_info<info::device::max_work_group_size>() /
          (*std::min_element(subgroups.begin(), subgroups.end())));

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        compile_num_sub_groups>) {
      return uint32_t{0};

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        max_sub_group_size>) {

      auto subgroups = dev.get_info<info::device::sub_group_sizes>();
      return static_cast<uint32_t>(
          *std::max_element(subgroups.begin(), subgroups.end()));

    } else if constexpr (std::is_same_v<KernelDeviceSpecificT,
                                        compile_sub_group_size>) {
      return uint32_t{0};
    } else {
      struct invalid_query{};
      return invalid_query{};
    }
  }

  template <typename Param>
  typename Param::return_type
  get_work_group_info(const device &dev) const;
};

HIPSYCL_SPECIALIZE_GET_INFO(kernel, function_name)
{ return "<implicitly mangled kernel name>"; }

HIPSYCL_SPECIALIZE_GET_INFO(kernel, num_args)
{ return 0; }

HIPSYCL_SPECIALIZE_GET_INFO(kernel, context)
{ return get_context(); }

HIPSYCL_SPECIALIZE_GET_INFO(kernel, program)
{ return get_program(); }

HIPSYCL_SPECIALIZE_GET_INFO(kernel, reference_count)
{ return 1; }

HIPSYCL_SPECIALIZE_GET_INFO(kernel, attributes)
{ return ""; }

#define HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(specialization)\
  template<> \
  inline typename info::kernel_work_group::specialization::return_type \
  kernel::get_work_group_info<info::kernel_work_group::specialization>(const device& dev) const

HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(global_work_size)
{
  // ToDO
  return range<3>{0,0,0};
}

HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(work_group_size)
{
  return dev.get_info<info::device::max_work_group_size>();
}

HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(compile_work_group_size)
{
  return range<3>{0,0,0};
}

HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(preferred_work_group_size_multiple)
{
  // ToDo
  return 128;
}

HIPSYCL_SPECIALIZE_KERNEL_GET_WORK_GROUP_INFO(private_mem_size)
{
  // ToDo
  return 0;
}

template <typename kernelT>
inline kernel program::get_kernel() const
{
  return kernel{"dummy-parameter", _ctx};
}

inline kernel program::get_kernel(string_class kernelName) const
{
  return kernel{"dummy-parameter", _ctx};
}

}
}

#endif
