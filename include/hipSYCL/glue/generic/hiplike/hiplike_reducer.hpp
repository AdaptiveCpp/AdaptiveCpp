/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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

#ifndef HIPSYCL_GLUE_HIPLIKE_REDUCER_HPP
#define HIPSYCL_GLUE_HIPLIKE_REDUCER_HPP

#include "hipSYCL/glue/generic/reduction_accumulator.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"

namespace hipsycl {
namespace glue {
namespace hiplike {

// For reductions with no known identity, stores whether any one thread has produced a value.
// TODO this is
//   a) somewhat wasteful with 7 unused bits of padding per flag
//   b) not friendly to (at least) CUDA shared memory where the smallest addressable unit is 32 bits
// We should evaluate whether bitmasks together with subgroup reductions can improve performance here.
using initialized_flag = bool;

// Provides a common interface to local_reducer for performing local-memory tree reduction on reduction operations
// with and without a known identity value.
template<class ReductionDescriptor, class Enable = void>
class local_reduction_accumulator;

// In the common case, when the identity of the reduction operation is known, each per-thread result can
// initialized to the identity value.
template<class ReductionDescriptor>
class local_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<ReductionDescriptor::has_identity>> {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;

  // scratch_values: local scratch memory, group-sized, uninitialized
  // output_values: local or global output value pointer, unit-sized, uninitialized
  // global_input_values: (optional) reduction input for stand-alone reduction kernels
  __host__ __device__ local_reduction_accumulator(value_type *scratch_values,
      value_type *output_values, value_type *global_input_values)
      : _scratch_values{scratch_values}, _output_values{output_values}, _global_input_values{global_input_values} {}

  // Initialize from a per-thread result
  __host__ __device__ void init_value(int my_lid,
      const private_accumulator_type &private_accumulator) {
    _scratch_values[my_lid] = private_accumulator.value;
  }

  // Combine intermediates in the same tree (for parallel local-memory reduction)
  __host__ __device__ void combine_with(const ReductionDescriptor &desc,
      int my_lid, int other_lid) {
    _scratch_values[my_lid] = desc.combiner(_scratch_values[my_lid], _scratch_values[other_lid]);
  }

  // Store intermediate output which will be re-loaded by a subsequent stand-alone reduction kernel
  __host__ __device__ void store_intermediate_output() {
    *_output_values = _scratch_values[0];
  }

  // Overwrite or combine reduction output into final SYCL output buffer
  __host__ __device__ void combine_final_output(const ReductionDescriptor &desc) {
     if (!desc.initialize_to_identity) {
       *_output_values = desc.combiner(*_output_values, _scratch_values[0]);
     } else {
       *_output_values = _scratch_values[0];
     }
  }

  // Construct a sequential accumulator from global input of a previous kernel
  __host__ __device__ private_accumulator_type get_global_input(int my_global_id) {
    return private_accumulator_type{_global_input_values[my_global_id]};
  }

private:
  value_type *_scratch_values;
  value_type *_output_values;
  value_type *_global_input_values;
};

// In the special case of a reduction operation without a known identity, we semantically reduce over optionals
// instead of values directly. Since not every item has to produce a value via combine(), the absence of inputs must
// be incorporated in the tree-reduction phase as well. Instead of using `std::optional` which will waste local memory
// on padding bytes for types with alignment > 1, we keep separate value-buffers and initialized-flag-buffers.
template<class ReductionDescriptor>
class local_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<!ReductionDescriptor::has_identity>> {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;

  // scratch_values: local scratch memory, group-sized, uninitialized
  // scratch_initialized_flags: local scratch memory, group-sized, uninitialized
  // output_values: local or global output value pointer, unit-sized, uninitialized
  // output_initialized_flags: pointer to flag in local or global memory, unit-sized, uninitialized
  // global_input_values: (optional) reduction input values for stand-alone reduction kernels
  // global_input_initialized_flags: (optional) reduction input initialized-flag for stand-alone reduction kernels
  __host__ __device__ local_reduction_accumulator(
      value_type *scratch_values, initialized_flag *scratch_initialized_flags,
      value_type *output_values, initialized_flag *output_initialized_flags,
      value_type *global_input_values, initialized_flag *global_input_initialized_flags)
    : _scratch_values{scratch_values}, _scratch_initialized_flags{scratch_initialized_flags}
    , _output_values{output_values}, _output_initialized_flags{output_initialized_flags}
    , _global_input_values{global_input_values}, _global_input_initialized_flags{global_input_initialized_flags} {}

  // Initialize from a per-thread result
  __host__ __device__ void init_value(int my_lid,
      const private_accumulator_type &private_accumulator) {
    _scratch_values[my_lid] = private_accumulator.value;
    _scratch_initialized_flags[my_lid] = private_accumulator.initialized;
  }

  // Combine intermediates in the same tree (for parallel local-memory reduction)
  __host__ __device__ void combine_with(const ReductionDescriptor &desc,
      int my_lid, int other_lid) {
    if (!_scratch_initialized_flags[other_lid]) return;
    _scratch_values[my_lid] = _scratch_initialized_flags[my_lid]
        ? desc.combiner(_scratch_values[my_lid], _scratch_values[other_lid])
        : _scratch_values[other_lid];
    _scratch_initialized_flags[my_lid] = true;
  }

  // Store intermediate output which will be re-loaded by a subsequent stand-alone reduction kernel
  __host__ __device__ void store_intermediate_output() {
    *_output_values = _scratch_values[0];
    *_output_initialized_flags = _scratch_initialized_flags[0];
  }

  // Overwrite or combine reduction output into final SYCL output buffer, if a value has been produced
  __host__ __device__ void combine_final_output(const ReductionDescriptor &desc) {
    if (_scratch_initialized_flags[0]) {
      *_output_values = desc.combiner(*_output_values, _scratch_values[0]);
    }
  }

  // Construct a sequential accumulator from global input of a previous kernel
  __host__ __device__ private_accumulator_type get_global_input_values(int my_global_id) {
    return private_accumulator_type{_global_input_values[my_global_id], _global_input_initialized_flags[my_global_id]};
  }

private:
  value_type *_scratch_values;
  initialized_flag *_scratch_initialized_flags;
  value_type *_output_values;
  initialized_flag *_output_initialized_flags;
  value_type *_global_input_values;
  initialized_flag *_global_input_initialized_flags;
};

template<class ReductionDescriptor>
class local_reducer {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using combiner_type = typename ReductionDescriptor::combiner_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;
  using local_accumulator_type = local_reduction_accumulator<ReductionDescriptor>;

  __host__ __device__ local_reducer(const ReductionDescriptor &desc, int my_lid,
                                    const local_accumulator_type &local_accumulator,
                                    bool is_final_stage)
      : _desc{desc}, _my_lid{my_lid},
        _private_accumulator{desc}, _local_accumulator{local_accumulator},
        _is_final_stage{is_final_stage} {}

  __host__ __device__
  void combine(const value_type& v) {
    _private_accumulator.combine_with(_desc, v);
  }

  __host__ __device__
  void finalize_result() {
    _local_accumulator.init_value(_my_lid, _private_accumulator);
    // TODO Optimize this - may be able to share code with group algorithms
    // TODO What if local size is not power of two?
#ifdef SYCL_DEVICE_ONLY
    __syncthreads();
    const int local_size =
        __hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z;
    for (int i = local_size / 2; i > 0; i /= 2) {
      if(_my_lid < i)
        _local_accumulator.combine_with(_desc, _my_lid, _my_lid + i);
      __syncthreads();
    }
    if (_my_lid == 0) {
      if (_is_final_stage) {
        _local_accumulator.combine_final_output(_desc);
      } else {
        _local_accumulator.store_intermediate_output();
      }
    }
#endif
  }

  __host__ __device__ void combine_global_input(int my_global_id) {
#ifdef SYCL_DEVICE_ONLY
    _private_accumulator.combine_with(_desc, _local_accumulator.get_global_input(my_global_id));
#endif
  }

  __host__ __device__ value_type identity() const {
    return _desc.identity;
  }

private:
  const ReductionDescriptor &_desc;
  const int _my_lid;
  private_accumulator_type _private_accumulator;
  local_accumulator_type _local_accumulator;
  bool _is_final_stage;
};

}
}
}

#endif