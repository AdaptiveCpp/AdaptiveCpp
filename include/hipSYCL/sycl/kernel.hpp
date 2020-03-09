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

#ifndef HIPSYCL_KERNEL_HPP
#define HIPSYCL_KERNEL_HPP

#include "context.hpp"
#include "program.hpp"
#include "device.hpp"
#include "info/info.hpp"

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
  
  template <info::kernel param>
  typename info::param_traits<info::kernel, param>::return_type 
  get_info() const;

  template <info::kernel_work_group param>
  typename info::param_traits<info::kernel_work_group, param>::return_type
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
  inline typename info::param_traits< \
        info::kernel_work_group, \
        info::kernel_work_group::specialization>::return_type \
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
