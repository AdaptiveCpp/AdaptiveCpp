/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay
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


#ifndef HIPSYCL_REDUCTION_GROUP_HORIZONTAL_REDUCER_HPP
#define HIPSYCL_REDUCTION_GROUP_HORIZONTAL_REDUCER_HPP

#include <vector>

#include "../reduction_descriptor.hpp"
#include "wi_reducer.hpp"
#include "wg_model_queries.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hipsycl::algorithms::reduction::wg_model {


/// Horizontal reducer for models where work groups exist
template<class GroupReductionAlgorithm>
class group_horizontal_reducer {
public:
  group_horizontal_reducer() = default;
  group_horizontal_reducer(const GroupReductionAlgorithm &grp_reduction_algorithm)
      : _group_reduction{grp_reduction_algorithm}{}

  template <class ReductionStage, typename... ReductionDescriptors>
  void configure_for_stage(ReductionStage &s, std::size_t stage_index,
                           std::size_t num_stages,
                           const ReductionDescriptors &...all_reductions) {
    _group_reduction.configure_for_stage(s, stage_index, num_stages,
                                         all_reductions...);
  }

  template <class ConfiguredReductionDescriptor>
  auto
  generate_wi_reducer(const ConfiguredReductionDescriptor &descriptor) const {
    return sequential_reducer<typename ConfiguredReductionDescriptor::op_type>{
        descriptor.get_operator()};
  }

  // Note: Assumes that all threads in the group enter this function!
  // kernel launchers need to take this into account when using the reduction engine
  // in range<>-based parallel_for, where typcally an if-statement
  // guards execution of the kernel function.
  template <class WiIndex, class ConfiguredReductionDescriptor, class WorkItemReducer>
  void finalize(const WiIndex &wi, const ConfiguredReductionDescriptor &descriptor,
                const WorkItemReducer &work_item_reducer) const {
    using value_type = typename ConfiguredReductionDescriptor::value_type;
    
    // To be set by _group_reduction
    bool is_leader;
    bool result_is_initialized;
    value_type group_result = _group_reduction(
        wi, descriptor, work_item_reducer, is_leader, result_is_initialized);
     
    if(is_leader) {
      if(descriptor.is_final_stage()) {
        reduction::detail::set_reduction_result(descriptor, group_result,
                                                result_is_initialized);
      } else {
        value_type* output = descriptor.get_stage_output();
        initialization_flag_t* is_initialized = descriptor.get_output_initialization_state();

        std::size_t group_id = get_group_linear_id(wi);
        output[group_id] = group_result;
        if constexpr (!ConfiguredReductionDescriptor::has_known_identity()){
          is_initialized[group_id] = result_is_initialized;
        }
      }
    }
  }

private:
  GroupReductionAlgorithm _group_reduction;
};



}

#endif
