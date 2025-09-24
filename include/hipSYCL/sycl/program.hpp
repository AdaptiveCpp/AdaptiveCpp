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

  program(const context &context, std::vector<device> deviceList)
  : _ctx{context}
  {}
  program(std::vector<program> programList, std::string linkOptions = ""){}

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
  void compile_with_kernel_type(std::string compileOptions = "")
  {}

  void compile_with_source(std::string kernelSource, std::string compileOptions = "")
  {}

  template <typename kernelT> void build_with_kernel_type(std::string buildOptions = "")
  {}

  void build_with_source(std::string kernelSource, std::string buildOptions = "")
  {
    // On CUDA, we may be able to use NVRTC library here for runtime compilation?
    throw exception{make_error_code(errc::feature_not_supported),
                    "program::build_with_source() is unimplemented."};
  }

  void link(std::string linkOptions = "")
  {}

  template <typename kernelT> 
  bool has_kernel() const
  { return true; }

  bool has_kernel(std::string kernelName) const
  { return true; }

  // get_kernel() is implemented in kernel.hpp.
  template <typename kernelT>
  kernel get_kernel() const;
  kernel get_kernel(std::string kernelName) const;

  template <typename Param>
  typename Param::return_type get_info() const;
  
  std::vector<std::vector<char>> get_binaries() const
  {
    return std::vector<std::vector<char>>{};
  }
  
  context get_context() const
  {
    return _ctx;
  }

  std::vector<device> get_devices() const;

  std::string get_compile_options() const
  { return ""; }

  std::string get_link_options() const
  { return ""; }

  std::string get_build_options() const
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
