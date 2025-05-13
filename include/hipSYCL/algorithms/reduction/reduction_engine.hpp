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
#ifndef HIPSYCL_REDUCTION_ENGINE_HPP
#define HIPSYCL_REDUCTION_ENGINE_HPP

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "hipSYCL/common/small_vector.hpp"

#include "hipSYCL/algorithms/util/allocation_cache.hpp"
#include "hipSYCL/sycl/libkernel/detail/data_layout.hpp"

#include "reduction_descriptor.hpp"
#include "reduction_plan.hpp"

#include "wg_model/wi_reducer.hpp"
#include "wg_model/wg_model_queries.hpp"
#include "wg_model/group_horizontal_reducer.hpp"
#include "wg_model/group_reduction_algorithms.hpp"
#include "wg_model/configured_reduction_descriptor.hpp"
#include "wg_model/reduction_stage.hpp"



namespace hipsycl::algorithms::reduction {





namespace detail {

template<class T>
T ceil_division(T a, T b) {
  return (a + b - 1) / b;
}

template<class FirstT, typename... Args>
FirstT get_first(FirstT&& f, Args&&... args) {
  return f;
}

template <class ReductionDescriptor>
auto configure_descriptor(
    const ReductionDescriptor &descriptor,
    const wg_model::reduction_stage_data &data_plan,
    std::size_t global_size) {
  using value_type = typename ReductionDescriptor::value_type;

  return wg_model::configured_reduction_descriptor<ReductionDescriptor>{
      descriptor,
      data_plan.is_input_initialized,
      data_plan.is_output_initialized,
      static_cast<value_type*>(data_plan.stage_input),
      static_cast<value_type*>(data_plan.stage_output),
      global_size};
}


template<class F, typename... Args>
static void enumerate_pack(F&& handler, Args&&... args) {
  std::size_t i = 0;
  ([&]() {
    handler(i, args);
    ++i;
  }(), ...);
}

template <class F, class ReductionStageType, std::size_t... Is,
          typename... ReductionDescriptors>
auto with_configured_descriptors(F &&f, std::index_sequence<Is...>,
                                 const ReductionStageType &stage_plan,
                                 ReductionDescriptors... descriptors) {
  return f(configure_descriptor<ReductionDescriptors>(
      descriptors, stage_plan.data_plan[Is], stage_plan.global_size)...);
}

template <class F, class ReductionStageType, typename... ReductionDescriptors>
auto with_configured_descriptors(F &&f, const ReductionStageType &stage_plan,
                                 ReductionDescriptors... descriptors) {
  return with_configured_descriptors(
      f, std::make_index_sequence<sizeof...(descriptors)>{}, stage_plan,
      descriptors...);
}
}


template<class GroupHorizontalReducer>
class wg_hierarchical_reduction_engine {
  GroupHorizontalReducer _reducer;
  util::allocation_group* _scratch_allocations;

  using reduction_stage_type = wg_model::reduction_stage<GroupHorizontalReducer>;

  static void determine_stages(
      std::size_t global_size, std::size_t wg_size,
      common::auto_small_vector<reduction_stage_type> &stages_out) {

    std::size_t current_num_groups = detail::ceil_division(global_size, wg_size);
    std::size_t current_num_work_items = global_size;

    stages_out.push_back(reduction_stage_type{
          wg_size, current_num_groups, current_num_work_items});

    while(current_num_groups > 1) {

      current_num_work_items = current_num_groups;
      current_num_groups = detail::ceil_division(current_num_groups, wg_size);
      // ToDo: Might want to use a different local size for pure
      // reduce steps
      stages_out.push_back(reduction_stage_type{
          wg_size, current_num_groups, current_num_work_items});
    }
  }

  template <class Kernel, typename... ConfiguredReductionDescriptors>
  static auto
  wrap_main_kernel(const Kernel &k, const GroupHorizontalReducer &group_reducer,
                   const ConfiguredReductionDescriptors &...descriptors) {

    auto with_unpacked_pack_by_value = [](auto f, auto... args) { f(args...); };

    auto wrapped_k = [=](auto... direct_kernel_args) {
      with_unpacked_pack_by_value(
          [=](auto... wi_reducers) {
            k(direct_kernel_args..., wi_reducers...);
            auto wi_index = detail::get_first(direct_kernel_args...);
            (group_reducer.finalize(wi_index, descriptors, wi_reducers), ...);
          },
          group_reducer.generate_wi_reducer(descriptors)...);
    };

    return wrapped_k;
  }

  template <class WiIndex, class ConfiguredReductionDescriptor>
  static void execute_dedicated_reduction(
      WiIndex wi_index, const GroupHorizontalReducer &group_reducer,
      const ConfiguredReductionDescriptor &descriptor) {
    
    auto wi_reducer = group_reducer.generate_wi_reducer(descriptor);
    std::size_t my_id = wg_model::get_global_linear_id(wi_index);
    
    if(my_id < descriptor.get_problem_size()) {
      if (descriptor.has_known_identity() ||
          descriptor.get_input_initialization_state()[my_id]) {
        wi_reducer.combine(descriptor.get_stage_input()[my_id]);
      }
    }
    group_reducer.finalize(wi_index, descriptor, wi_reducer);
  }

  template <typename... ConfiguredReductionDescriptors>
  static auto create_dedicated_reduction_kernel(
      const GroupHorizontalReducer &group_reducer,
      const ConfiguredReductionDescriptors &...inputs) {

    auto kernel = [=](auto first_arg, auto &&...direct_kernel_args) {
      (execute_dedicated_reduction(first_arg, group_reducer, inputs), ...);
    };

    return kernel;
  }

  
  template <class Kernel, class PlanType, typename... ReductionDescriptors>
  auto make_main_reducing_kernel(Kernel main_kernel,
                                 const PlanType &reduction_plan,
                                 ReductionDescriptors... descriptors) {
    assert(reduction_plan.size() > 0);

    return detail::with_configured_descriptors(
        [=, &reduction_plan](auto... configured_descriptors) {
          
          auto horizontal_reducer = reduction_plan[0].reducer;

          return wrap_main_kernel(main_kernel, horizontal_reducer,
                                          configured_descriptors...);
        },
        reduction_plan[0], descriptors...);
  }

  template <class KernelLauncher, class PlanType,
            typename... ReductionDescriptors>
  void run_additional_kernels(KernelLauncher kernel_launcher,
                              const PlanType &reduction_plan,
                              ReductionDescriptors... descriptors) {

    // start from 1 to skip the primary kernel
    for(int i = 1; i < reduction_plan.size(); ++i) {
      auto kernel = detail::with_configured_descriptors(
        [=, &reduction_plan](auto... configured_descriptors) {
          
          auto horizontal_reducer = reduction_plan[i].reducer;

          return create_dedicated_reduction_kernel(
              horizontal_reducer, configured_descriptors...);
        }, reduction_plan[i], descriptors...);

      kernel_launcher(reduction_plan[i].num_groups, reduction_plan[i].wg_size,
                      reduction_plan[i].global_size,
                      reduction_plan[i].local_mem, kernel);
    }
  }
public:
  wg_hierarchical_reduction_engine(
      const GroupHorizontalReducer &horizontal_reducer,
      util::allocation_group *scratch_allocation_group)
      : _scratch_allocations{scratch_allocation_group},
        _reducer{horizontal_reducer} {}


  /// Create reduction plan.
  /// Even in basic parallel for, a work group size must be provided
  /// on models where a group size size is availalbe (such as on device)
  /// such that the reduction driver knows the number of group outputs
  /// of the first reduction stage.
  /// In models where work group size genuinely does not matter
  /// (like CPU basic parallel_for), wg_size should be set to 1.
  template <typename... ReductionDescriptors>
  reduction_plan<reduction_stage_type, ReductionDescriptors...>
  create_plan(std::size_t global_size, std::size_t wg_size,
              ReductionDescriptors... descriptors) const {

    assert(wg_size > 0);

    std::size_t reduction_wg_size = wg_size;
    if(reduction_wg_size <= 1)
      reduction_wg_size = 128;

    reduction_plan<reduction_stage_type, ReductionDescriptors...> result_plan{
        descriptors...};

    // Add Primary stage
    if(wg_size == 0)
      wg_size = 1;
    
    result_plan.push_back(reduction_stage_type{
        wg_size, detail::ceil_division(global_size, wg_size), global_size});
    
    // Give reducer the chance to perform its own stage calculation
    common::auto_small_vector<reduction_stage_type> additional_plan;
    
    std::size_t num_groups = detail::ceil_division(global_size, wg_size);
    // if we only have a single group, we are already done.
    if(num_groups > 1)
      determine_stages(num_groups, reduction_wg_size, additional_plan);
  

    for(const auto& stage : additional_plan) {
      result_plan.push_back(stage);
    }
    
    // Then, allocate required scratch and plan scratch data usage.
    const std::size_t num_reductions = sizeof...(ReductionDescriptors); 
    // Initialize all to nullptr
    for(auto& stage : result_plan) {
      stage.data_plan.resize(num_reductions);
      for(std::size_t reduction = 0; reduction < num_reductions; ++reduction) {
        stage.data_plan[reduction].is_input_initialized = nullptr;
        stage.data_plan[reduction].is_output_initialized = nullptr;
        stage.data_plan[reduction].stage_input = nullptr;
        stage.data_plan[reduction].stage_output = nullptr;
      }
    }
    
    // If we only need the main kernel for the reduction, no scratch is needed.
    if(result_plan.size() > 1) {
      detail::enumerate_pack(
          [&](std::size_t reduction_index, const auto &descriptor) {
            bool has_known_identity = descriptor.has_known_identity();

            using value_type =
                typename std::decay_t<decltype(descriptor)>::value_type;

            value_type *stage_scratch_a = nullptr;
            value_type *stage_scratch_b = nullptr;
            initialization_flag_t *is_initialized_a = nullptr;
            initialization_flag_t *is_initialized_b = nullptr;

            stage_scratch_a = _scratch_allocations->obtain<value_type>(
                result_plan[1].global_size);
            if (result_plan.size() > 1)
              stage_scratch_b = _scratch_allocations->obtain<value_type>(
                  result_plan[1].global_size);

            if (!has_known_identity) {
              is_initialized_a =
                  _scratch_allocations->obtain<initialization_flag_t>(
                      result_plan[1].global_size);
              if (result_plan.size() > 1)
                is_initialized_b =
                    _scratch_allocations->obtain<initialization_flag_t>(
                        result_plan[1].global_size);
            }

            for (std::size_t i = 0; i < result_plan.size(); ++i) {
              auto &stage = result_plan[i];
              // inputs should remain nullptr for first stage
              if (i > 0) {
                // Use B scratch, because B is only allocated if we have
                // intermediate stages, while A is available always.
                stage.data_plan[reduction_index].stage_input = stage_scratch_b;
                stage.data_plan[reduction_index].is_input_initialized =
                    is_initialized_b;
              }
              // outputs should be set to nullptr for last stage
              if (i != result_plan.size() - 1) {
                stage.data_plan[reduction_index].stage_output = stage_scratch_a;
                stage.data_plan[reduction_index].is_output_initialized =
                    is_initialized_a;
              } else {
                stage.data_plan[reduction_index].stage_output =
                    nullptr;
                stage.data_plan[reduction_index].is_output_initialized =
                    nullptr;
              }
              std::swap(stage_scratch_a, stage_scratch_b);
              std::swap(is_initialized_a, is_initialized_b);
            }
          },
          descriptors...);
    }

    // Lastly, configure reducers in each stage based on the
    // value of _reducer.
    for(std::size_t i = 0; i < result_plan.size(); ++i) {
      auto stage_reducer = _reducer;
      result_plan[i].reducer = _reducer;
      result_plan[i].reducer.configure_for_stage(
          result_plan[i], i, result_plan.size(), descriptors...);
    }

    return result_plan;
  }

  template <class Kernel, class PlanType, typename... ReductionDescriptors>
  auto make_main_reducing_kernel(Kernel main_kernel,
                                 const PlanType &reduction_plan) {
    assert(reduction_plan.size() > 0);

    return std::apply(
        [&](const auto&... descriptors) {
          return this->make_main_reducing_kernel(main_kernel, reduction_plan,
                                                 descriptors...);
        },
        reduction_plan.get_descriptors());
  }

  template <class KernelLauncher, class PlanType,
            typename... ReductionDescriptors>
  void run_additional_kernels(KernelLauncher kernel_launcher,
                              const PlanType &reduction_plan) {

    std::apply(
        [&](const auto&... descriptors) {
          this->run_additional_kernels(kernel_launcher, reduction_plan,
                                       descriptors...);
        },
        reduction_plan.get_descriptors());
  }
};


}

#endif
