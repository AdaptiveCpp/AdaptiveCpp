/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
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

#ifndef HIPSYCL_ADAPTIVITY_ENGINE_HPP
#define HIPSYCL_ADAPTIVITY_ENGINE_HPP


#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/runtime/util.hpp"
#include "hipSYCL/runtime/kernel_cache.hpp"

namespace hipsycl {
namespace rt {

class kernel_adaptivity_engine {
public:
  kernel_adaptivity_engine(
    hcf_object_id hcf_object,
    const std::string& backend_kernel_name,
    const hcf_kernel_info* kernel_info,
    const glue::jit::cxx_argument_mapper& arg_mapper,
    const range<3>& num_groups,
    const range<3>& block_size,
    void** args,
    std::size_t* arg_sizes,
    std::size_t num_args,
    std::size_t local_mem_size);

  kernel_configuration::id_type
  finalize_binary_configuration(kernel_configuration &config);

  std::string select_image_and_kernels(std::vector<std::string>* kernel_names_out);
private:
  hcf_object_id _hcf;
  const std::string& _kernel_name;
  const hcf_kernel_info* _kernel_info;
  const glue::jit::cxx_argument_mapper& _arg_mapper;
  const range<3>& _num_groups;
  const range<3>& _block_size;
  void** _args;
  std::size_t* _arg_sizes;
  std::size_t _num_args;
  std::size_t _local_mem_size;

  int _adaptivity_level;
};

}
}

#endif
