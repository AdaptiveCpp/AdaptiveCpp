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
