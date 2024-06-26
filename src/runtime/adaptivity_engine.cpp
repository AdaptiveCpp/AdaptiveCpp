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
#include <limits>


namespace hipsycl {
namespace rt {

namespace {

template<class F>
void access_appdb(common::db::appdb& db, bool needs_write_access, F&& handler) {
  if(needs_write_access) {
    db.read_write_access(handler);
  } else {
    db.read_access(handler);
  }
}

bool has_annotation(const hcf_kernel_info *info, int param_index,
                    hcf_kernel_info::annotation_type annotation) {
  for(auto a : info->get_known_annotations(param_index)) {
    if(a == annotation)
      return true;
  }
  return false;
}

// Estimates whether kernel arguments might be invariant. This also updates
// the statistics in the appdb, so if this function returns true,
// a specialization should be carried out by the calling code in order
// to ensure consistency of the appdb with what is actually happening.
bool is_likely_invariant_argument(common::db::kernel_entry &kernel_entry,
                                  int param_index, std::size_t application_run,
                                  uint64_t current_value) {
  auto& args = kernel_entry.kernel_args;

  const int static_specialization_trigger =
      application::get_settings().get<setting::jitopt_iads_static_trigger>();
  const double relative_specialization_trigger =
      application::get_settings().get<setting::jitopt_iads_relative_trigger>();

  // In case we find an empty slot, this stores its index.
  int empty_slot = -1;

  for(int i = 0; i < common::db::kernel_arg_entry::max_tracked_values; ++i) {
    // How many times the current kernel parameter was set to
    // args[param_index].common_values[i]

    auto& arg_statistics = args[param_index].common_values[i];
    uint64_t& arg_value_count = arg_statistics.count;
    // Is the argument the same as an argument from a previous submission that we
    // are tracking as commonly used?
    if(arg_statistics.value == current_value && arg_value_count > 0) {
      // Yep, we've hit it again, increase counter
      ++arg_value_count;
      arg_statistics.last_used = kernel_entry.num_registered_invocations;

      bool& is_already_specialized = args[param_index].was_specialized[i];
      // If we already have specialized in the past, continue to specialize.
      // This prevents performance regressions if the first the value is specialized,
      // then not used for a long while and we are now seeing it again.
      if(is_already_specialized)
        return true;

      double fraction_of_all_invocations = static_cast<double>(arg_value_count) /
               kernel_entry.num_registered_invocations;

      if (arg_value_count > static_specialization_trigger ||
          ((fraction_of_all_invocations > relative_specialization_trigger) &&
           application_run > 0)) {
        is_already_specialized = true;
        return true;
      } else
        return false;
    } else if(arg_value_count == 0) {
      // Remember that we have hit an unused slot in case we don't find any
      // matches with values that are know to be commonly occuring
      empty_slot = i; 
    }
  }

  auto create_new_entry = [&](int slot_index) {
    common::db::kernel_arg_value_statistics new_arg_entry;
    new_arg_entry.value = current_value;
    new_arg_entry.count = 1;
    new_arg_entry.last_used = kernel_entry.num_registered_invocations;
    args[param_index].common_values[slot_index] = new_arg_entry;
    args[param_index].was_specialized[slot_index] = false;
  };

  // If we arrive here, we are dealing with a value that we have
  // not encountered before.
  if(empty_slot >= 0) {
    // If we have an empty slot, store the current argument in case
    // it gets used a lot by future kernel invocations.
    create_new_entry(empty_slot);
  } else {
    // Try to find an old entry that we can evict.
    int eviction_candidate_slot = -1;
    uint64_t eviction_candidate_last_used_time = std::numeric_limits<uint64_t>::max();

    for(int i = 0; i < common::db::kernel_arg_entry::max_tracked_values; ++i) {
      auto& arg_statistics = args[param_index].common_values[i];
      
      if(arg_statistics.last_used < eviction_candidate_last_used_time) {
        auto age = kernel_entry.num_registered_invocations - arg_statistics.last_used;
        double fraction_of_all_invocations = static_cast<double>(arg_statistics.count) /
               kernel_entry.num_registered_invocations;

        if (age > static_specialization_trigger &&
            fraction_of_all_invocations < relative_specialization_trigger) {
          
          // Update least-recently-used so that we can find potential entries to evict
          eviction_candidate_slot = i;
          eviction_candidate_last_used_time = arg_statistics.last_used;
        }
      }
    }
    
    if (eviction_candidate_slot >= 0) {
      create_new_entry(eviction_candidate_slot);
    }
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
      std::size_t arg_size = _kernel_info->get_argument_size(i);
      if (has_annotation(_kernel_info, i,
                         hcf_kernel_info::annotation_type::specialized) &&
          arg_size <= sizeof(uint64_t)) {
        uint64_t buffer_value = 0;
        std::memcpy(&buffer_value, _arg_mapper.get_mapped_args()[i], arg_size);
        config.set_specialized_kernel_argument(i, buffer_value);
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
        if (_kernel_info->get_argument_type(i) !=
                hcf_kernel_info::argument_type::pointer &&
            is_likely_invariant_argument(kernel_entry, i, data.content_version,
                                         arg_value) &&
            !has_annotation(_kernel_info, i,
                            hcf_kernel_info::annotation_type::specialized)) {
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