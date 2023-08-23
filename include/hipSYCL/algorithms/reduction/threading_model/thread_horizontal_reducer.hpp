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


#ifndef HIPSYCL_REDUCTION_THREAD_HORIZONTAL_REDUCER_HPP
#define HIPSYCL_REDUCTION_THREAD_HORIZONTAL_REDUCER_HPP

#include <vector>

#include "../reduction_descriptor.hpp"
#include "wi_reducer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace hipsycl::algorithms::reduction::threading_model {

class omp_thread_info_query {
public:
  int get_max_num_threads() const noexcept {
#ifdef _OPENMP
    __hipsycl_if_target_host(
      return omp_get_max_threads();
    );
    __hipsycl_if_target_device(
      return 1;
    );
#else
    return 1;
#endif
  }

  int get_my_thread_id() const noexcept {
#ifdef _OPENMP
    __hipsycl_if_target_host(
      return omp_get_thread_num();
    )
    __hipsycl_if_target_device(
      return 0;
    )
#else
    return 0;
#endif
  }

  int get_num_threads() const noexcept {
#ifdef _OPENMP
    __hipsycl_if_target_host(
      return omp_get_num_threads();
    );
    __hipsycl_if_target_device(
      return 1;
    );
#else
    return 1;
#endif
  }
};

/// Horizontal reducer for models on the host where work groups don't exist
/// This reducer works a bit differently: Each thread directly accumulates
/// into the entry corresponding to the thread id in the stage output.
///
/// The reduction model here is based on the assumption of a strict two-stage setup:
/// 1. A multi-threaded stage using thread_horizontal_reducer
/// 2. A single_task stage gathering all the values.
/// When using this model, you *MUST* pass the reduction a driver a single-task
/// kernel-launcher that iterates over all the entries!
///
/// TODO: It's unclear whether it's a good idea to shove this design with all its
/// assumptions into the same reduction_driver implementation that is also used
/// for the work-group-based reductions. Would it be better to have an entirely separate
/// reduction_driver implementation for thread_horizontal_reducer?
template<class ThreadInfoQuery = omp_thread_info_query>
class thread_horizontal_reducer {
public:
  thread_horizontal_reducer() = default;
  thread_horizontal_reducer(const ThreadInfoQuery &info_query)
      : _info_query{info_query}{}

  template <class ConfiguredReductionDescriptor>
  auto
  generate_wi_reducer(const ConfiguredReductionDescriptor &descriptor) const {

    int my_thread_id = _info_query.get_my_thread_id();
    auto* stage_output = descriptor.get_scratch() + my_thread_id;
    auto* initialization_state = descriptor.get_initialization_state() + my_thread_id;

    return threading_model::sequential_reducer<
        typename ConfiguredReductionDescriptor::op_type>{
        descriptor.get_operator(), stage_output, initialization_state};
  }

  // Note: Assumes that all threads in the group enter this function!
  // kernel launchers need to take this into account when using the reduction engine
  // in range<>-based parallel_for, where typcally an if-statement
  // guards execution of the kernel function.
  template <class WiIndex, class ConfiguredReductionDescriptor, class WorkItemReducer>
  void finalize(const WiIndex &wi, const ConfiguredReductionDescriptor &descriptor,
                const WorkItemReducer &work_item_reducer) const {
    // Work item reducer directly writes into the stage output, so nothing to do here anymore.
  }

private:
  ThreadInfoQuery _info_query;

};

}

#endif
