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
#include "hipSYCL/common/appdb.hpp"
#include "hipSYCL/runtime/kernel_configuration.hpp"
#include "hipSYCL/glue/llvm-sscp/jit.hpp"
#include "hipSYCL/runtime/application.hpp"
#include "hipSYCL/common/filesystem.hpp"


namespace hipsycl {
namespace rt {

namespace {

bool is_likely_invariant_argument(common::db::kernel_entry &kernel_entry,
                                  int param_index, std::size_t application_run,
                                  uint64_t current_value) {
  auto& args = kernel_entry.kernel_args;
  
  // In case we find an empty slot, this stores its index.
  int empty_slot = -1;
  for(int i = 0; i < common::db::kernel_arg_entry::max_tracked_values; ++i) {
    // How many times the current kernel parameter was set to
    // args[param_index].common_values[i]
    std::size_t& arg_value_count = args[param_index].common_values_count[i];
    // Is the argument the same as an argument from a previous submission that we
    // are tracking as commonly used?
    if(args[param_index].common_values[i] == current_value && arg_value_count > 0) {
      // Yep, we've hit it again, increase counter
      ++arg_value_count;

      double fraction_of_all_invocations = static_cast<double>(arg_value_count) /
               kernel_entry.num_registered_invocations;

      if (arg_value_count > 128 ||
          ((fraction_of_all_invocations >
           0.8) && application_run > 0))
        return true;
      else
        return false;
    } else if(arg_value_count == 0) {
      // Remember that we have hit an unused slot in case we don't find any
      // matches with values that are know to be commonly occuring
      empty_slot = i; 
    }
  }

  if(empty_slot >= 0) {
    // If we have an empty slot, store the current argument in case
    // it gets used a lot by future kernel invocations.
    args[param_index].common_values_count[empty_slot] = 1;
    args[param_index].common_values[empty_slot] = current_value;
  }

  return false;
}
}

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

  if(_adaptivity_level > 1) {
    auto base_id = config.generate_id();
    
    // Automatic application of specialization constants by detecting
    // invariant kernel arguments
    auto& appdb = common::filesystem::persistent_storage::get().get_this_app_db();
    appdb.read_write_access([&](common::db::appdb_data& data){
      auto& kernel_entry = data.kernels[base_id];
      ++kernel_entry.num_registered_invocations;

      std::size_t num_kernel_args = _kernel_info->get_num_parameters();
      if(kernel_entry.kernel_args.size() != num_kernel_args)
        kernel_entry.kernel_args.resize(num_kernel_args);

      for(int i = 0; i < num_kernel_args; ++i) {
        uint64_t arg_value = 0;
        std::memcpy(&arg_value, _arg_mapper.get_mapped_args()[i],
                    _kernel_info->get_argument_size(i));
        // TODO: Don't specialize if specialized<> is already used
        if(_kernel_info->get_argument_type(i) != hcf_kernel_info::argument_type::pointer &&
          is_likely_invariant_argument(kernel_entry, i, data.content_version, arg_value)) {
          HIPSYCL_DEBUG_INFO << "adaptivity_engine: Kernel argument " << i
                             << " is invariant or common, specializing."
                             << std::endl;
          config.set_specialized_kernel_argument(i, arg_value);
        } else {
          HIPSYCL_DEBUG_INFO << "adaptivity_engine: Not specializing kernel argument " << i
                             << std::endl;
        }
      }
    });
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