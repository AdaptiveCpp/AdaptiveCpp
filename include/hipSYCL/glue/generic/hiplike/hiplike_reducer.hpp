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

using local_memory_flag = bool;

template<class ReductionDescriptor, class Enable = void>
class local_reduction_accumulator;

template<class ReductionDescriptor>
class local_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<ReductionDescriptor::has_identity>> {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;

  __host__ __device__ local_reduction_accumulator(value_type *local_memory,
      value_type *local_output, value_type *global_input)
      : _local_memory{local_memory}, _local_output{local_output}, _global_input{global_input} {}

  __host__ __device__ void init_value(int my_lid,
      const private_accumulator_type &private_accumulator) {
    _local_memory[my_lid] = private_accumulator.value;
  }

  __host__ __device__ void combine_with(const ReductionDescriptor &desc,
      int my_lid, int other_lid) {
    _local_memory[my_lid] = desc.combiner(_local_memory[my_lid], _local_memory[other_lid]);
  }

  __host__ __device__ void store_intermediate_output() {
    *_local_output = _local_memory[0];
  }

  __host__ __device__ void combine_final_output(const ReductionDescriptor &desc) {
     if (!desc.initialize_to_identity) {
       *_local_output = desc.combiner(*_local_output, _local_memory[0]);
     } else {
       *_local_output = _local_memory[0];
     }
  }

  __host__ __device__ private_accumulator_type get_global_input(int my_global_id) {
    return private_accumulator_type{_global_input[my_global_id]};
  }

private:
  value_type *_local_memory;
  value_type *_local_output;
  value_type *_global_input;
};

template<class ReductionDescriptor>
class local_reduction_accumulator<ReductionDescriptor,
    std::enable_if_t<!ReductionDescriptor::has_identity>> {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;

  __host__ __device__ local_reduction_accumulator(
      value_type *local_memory, local_memory_flag *memory_initialized,
      value_type *local_output, local_memory_flag *output_initialized,
      value_type *global_input, local_memory_flag *input_initialized)
    : _local_memory{local_memory}, _memory_initialized{memory_initialized}
    , _local_output{local_output}, _output_initialized{output_initialized}
    , _global_input{global_input}, _input_initialized{input_initialized} {}

  __host__ __device__ void init_value(int my_lid,
      const private_accumulator_type &private_accumulator) {
    _local_memory[my_lid] = private_accumulator.value;
    _memory_initialized[my_lid] = private_accumulator.initialized;
  }

  __host__ __device__ void combine_with(const ReductionDescriptor &desc,
      int my_lid, int other_lid) {
    if (!_memory_initialized[other_lid]) return;
    _local_memory[my_lid] = _memory_initialized[my_lid]
        ? desc.combiner(_local_memory[my_lid], _local_memory[other_lid])
        : _local_memory[other_lid];
    _memory_initialized[my_lid] = true;
  }

  __host__ __device__ void store_intermediate_output() {
    *_local_output = _local_memory[0];
    *_output_initialized = _memory_initialized[0];
  }

  __host__ __device__ void combine_final_output(const ReductionDescriptor &desc) {
    if (_memory_initialized[0]) {
      *_local_output = desc.combiner(*_local_output, _local_memory[0]);
    }
  }

  __host__ __device__ private_accumulator_type get_global_input(int my_global_id) {
    return private_accumulator_type{_global_input[my_global_id], _input_initialized[my_global_id]};
  }

private:
  value_type *_local_memory;
  local_memory_flag *_memory_initialized;
  value_type *_local_output;
  local_memory_flag *_output_initialized;
  value_type *_global_input;
  local_memory_flag *_input_initialized;
};

template<class ReductionDescriptor>
class local_reducer {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using combiner_type = typename ReductionDescriptor::combiner_type;
  using private_accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;
  using local_accumulator_type = local_reduction_accumulator<ReductionDescriptor>;

  __host__ __device__ local_reducer(const ReductionDescriptor &desc, int my_lid,
                                    const private_accumulator_type &private_accumulator,
                                    const local_accumulator_type &local_accumulator,
                                    bool is_final_stage)
      : _desc{desc}, _my_lid{my_lid},
        _private_accumulator{private_accumulator}, _local_accumulator{local_accumulator},
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