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

#include "hipSYCL/runtime/adaptivity_engine.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/application.hpp"


namespace hipsycl {
namespace rt {

kernel_adaptivity_engine::kernel_adaptivity_engine(
    hcf_object_id hcf_object, const std::string &backend_kernel_name,
    const hcf_kernel_info *kernel_info,
    const glue::jit::cxx_argument_mapper &arg_mapper,
    const range<3> &num_groups, const range<3> &block_size, void **args,
    std::size_t *arg_sizes, std::size_t num_args, std::size_t local_mem_size)
    : _hcf{hcf_object}, _kernel_name{backend_kernel_name},
      _kernel_info{kernel_info}, _arg_mapper{arg_mapper},
      _num_groups{num_groups}, _block_size{block_size}, _args{args},
      _arg_sizes{arg_sizes}, _num_args{num_args},
      _local_mem_size(local_mem_size) {

  _adaptivity_level = application::get_settings().get<setting::adaptivity_level>();
}

kernel_configuration::id_type
kernel_adaptivity_engine::finalize_binary_configuration(
    kernel_configuration &config) {

  if(_adaptivity_level > 0) {
    // Enter single-kernel code model
    config.append_base_configuration(
        kernel_base_config_parameter::single_kernel, _kernel_name);

    // Hard-code group sizes into the JIT binary
    config.set_build_option(kernel_build_option::known_group_size_x,
                            _block_size[0]);
    config.set_build_option(kernel_build_option::known_group_size_y,
                            _block_size[1]);
    config.set_build_option(kernel_build_option::known_group_size_z,
                            _block_size[2]);

    // Try to optimize size_t -> i32 for queries if those fit in int
    auto global_size = _num_groups * _block_size;
    auto int_max = std::numeric_limits<int>::max();
    if (global_size[0] * global_size[1] * global_size[2] < int_max)
      config.set_build_flag(kernel_build_flag::global_sizes_fit_in_int);

    // Hard-code local memory size into the JIT binary
    config.set_build_option(kernel_build_option::known_local_mem_size,
                            _local_mem_size);

    // Handle kernel parameter optimization hints
    for(int i = 0; i < _kernel_info->get_num_parameters(); ++i) {
      auto& annotations = _kernel_info->get_known_annotations(i);
      std::size_t arg_size = _kernel_info->get_argument_size(i);
      for(auto annotation : annotations) {
        if (annotation == hcf_kernel_info::annotation_type::specialized &&
            arg_size <= sizeof(uint64_t)) {
          uint64_t buffer_value = 0;
          std::memcpy(&buffer_value, _arg_mapper.get_mapped_args()[i], arg_size);
          config.set_specialized_kernel_argument(i, buffer_value);
        }
      }
    }
  }

  return config.generate_id();
}

std::string kernel_adaptivity_engine::select_image_and_kernels(std::vector<std::string>* kernel_names_out){
  if(_adaptivity_level > 0) {
    *kernel_names_out = std::vector{_kernel_name};

    std::vector<std::string> all_kernels_in_image;
    return  glue::jit::select_image(_kernel_info, &all_kernels_in_image);
  } else {
    return glue::jit::select_image(_kernel_info, kernel_names_out);
  }
}

}
}