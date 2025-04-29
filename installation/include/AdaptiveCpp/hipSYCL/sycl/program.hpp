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
#ifndef HIPSYCL_PROGRAM_HPP
#define HIPSYCL_PROGRAM_HPP

#include "types.hpp"
#include "context.hpp"
#include "exception.hpp"
#include "info/info.hpp"

namespace hipsycl {
namespace sycl {

enum class program_state 
{
  none,
  compiled,
  linked
};

class kernel;

// Dummy implementation of SYCL program class
class program 
{
  context _ctx;
public:
  program() = delete;
  
  explicit program(const context &context)
  : _ctx{context}
  {}

  program(const context &context, vector_class<device> deviceList)
  : _ctx{context}
  {}
  program(vector_class<program> programList, string_class linkOptions = ""){}

  template<class Cl_program>
  program(const context &context, Cl_program clProgram)
  : _ctx{context}
  {}
  
  /* -- common interface members -- */
  //cl_program get() const;

  bool is_host() const
  {
    return _ctx.is_host();
  }
  
  template <typename kernelT>
  void compile_with_kernel_type(string_class compileOptions = "")
  {}

  void compile_with_source(string_class kernelSource, string_class compileOptions = "")
  {}

  template <typename kernelT> void build_with_kernel_type(string_class buildOptions = "")
  {}

  void build_with_source(string_class kernelSource, string_class buildOptions = "")
  {
    // On CUDA, we may be able to use NVRTC library here for runtime compilation?
    throw exception{make_error_code(errc::feature_not_supported),
                    "program::build_with_source() is unimplemented."};
  }

  void link(string_class linkOptions = "")
  {}

  template <typename kernelT> 
  bool has_kernel() const
  { return true; }

  bool has_kernel(string_class kernelName) const
  { return true; }

  // get_kernel() is implemented in kernel.hpp.
  template <typename kernelT>
  kernel get_kernel() const;
  kernel get_kernel(string_class kernelName) const;

  template <typename Param>
  typename Param::return_type get_info() const;
  
  vector_class<vector_class<char>> get_binaries() const
  {
    return vector_class<vector_class<char>>{};
  }
  
  context get_context() const
  {
    return _ctx;
  }

  vector_class<device> get_devices() const;

  string_class get_compile_options() const
  { return ""; }

  string_class get_link_options() const
  { return ""; }

  string_class get_build_options() const
  { return ""; }

  program_state get_state() const
  {
    return program_state::linked;
  }
};

HIPSYCL_SPECIALIZE_GET_INFO(program, reference_count)
{
  return 1;
}

HIPSYCL_SPECIALIZE_GET_INFO(program, context)
{
  return get_context();
}

HIPSYCL_SPECIALIZE_GET_INFO(program, devices)
{
  return get_context().get_devices();
}

}
}

#endif
