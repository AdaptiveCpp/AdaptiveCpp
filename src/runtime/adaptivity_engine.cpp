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
#include "hipSYCL/runtime/application.hpp"

namespace hipsycl {
namespace rt {

namespace {

std::string group_size_build_opt_x = "known-group-size-x";
std::string group_size_build_opt_y = "known-group-size-y";
std::string group_size_build_opt_z = "known-group-size-z";

}

kernel_adaptivity_engine::kernel_adaptivity_engine(hcf_object_id hcf_object,
                                     const std::string &backend_kernel_name,
                                     const hcf_kernel_info* kernel_info,
                                     const range<3> &num_groups,
                                     const range<3> &block_size, void **args,
                                     std::size_t *arg_sizes,
                                     std::size_t num_args)
    : _hcf{hcf_object}, _kernel_name{backend_kernel_name}, _kernel_info{kernel_info},
      _num_groups{num_groups}, _block_size{block_size}, _args{args},
      _arg_sizes{arg_sizes}, _num_args{num_args} {

  _adaptivity_level = application::get_settings().get<setting::adaptivity_level>();
}

glue::kernel_configuration::id_type
kernel_adaptivity_engine::finalize_binary_configuration(
    glue::kernel_configuration &config) {

  if(_adaptivity_level > 0) {
    config.set_build_option(group_size_build_opt_x,
                            std::to_string(_block_size[0]));
    config.set_build_option(group_size_build_opt_x,
                            std::to_string(_block_size[1]));
    config.set_build_option(group_size_build_opt_x,
                            std::to_string(_block_size[2]));
  }

  return config.generate_id();
}

std::vector<std::string> kernel_adaptivity_engine::get_target_kernels() {
  if(_adaptivity_level > 0) {
    return std::vector{_kernel_name};
  }
}

}
}