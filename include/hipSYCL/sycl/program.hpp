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
    throw unimplemented{"program::build_with_source() is unimplemented."};
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

  template <info::program param> typename info::param_traits<info::program, param>::return_type
  get_info() const;
  
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
