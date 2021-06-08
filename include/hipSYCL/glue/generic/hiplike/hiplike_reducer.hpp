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

#include "hipSYCL/sycl/libkernel/backend.hpp"

namespace hipsycl {
namespace glue {
namespace hiplike {

template<class ReductionDescriptor, class Enable = void>
class local_reducer;

template<class ReductionDescriptor>
class local_reducer<ReductionDescriptor, std::enable_if_t<ReductionDescriptor::has_identity>> {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using combiner_type = typename ReductionDescriptor::combiner_type;

  __host__ __device__ local_reducer(const ReductionDescriptor &desc, int my_lid,
                                    value_type *local_memory,
                                    value_type *local_output,
                                    value_type *global_input, bool is_final_stage)
      : _desc{desc}, _my_lid{my_lid}, _my_value{desc.identity},
        _local_memory{local_memory}, _local_output{local_output},
        _global_input{global_input}, _is_final_stage{is_final_stage} {}

  __host__ __device__
  void combine(const value_type& v) {
    _my_value = _desc.combiner(_my_value, v);
  }

  __host__ __device__
  void finalize_result() {
    _local_memory[_my_lid] = _my_value;
    // TODO Optimize this - may be able to share
    // code with group algorithms
    // TODO What if local size is not power of two?
#ifdef SYCL_DEVICE_ONLY
    __syncthreads();
    const int local_size =
        __hipsycl_lsize_x * __hipsycl_lsize_y * __hipsycl_lsize_z;
    for (int i = local_size / 2; i > 0; i /= 2) {
      if(_my_lid < i)
        _local_memory[_my_lid] =
            _desc.combiner(_local_memory[_my_lid], _local_memory[_my_lid + i]);
      __syncthreads();
    }
    if (_my_lid == 0) {
      if (_is_final_stage && !_desc.initialize_to_identity) {
        *_local_output = _desc.combiner(*_local_output, _local_memory[0]);
      } else {
        *_local_output = _local_memory[0];
      }
    }
#endif
  }

  __host__ __device__ void combine_global_input(int my_global_id) {
#ifdef SYCL_DEVICE_ONLY
    combine(_global_input[my_global_id]);
#endif
  }
private:
  const ReductionDescriptor &_desc;
  const int _my_lid;
  value_type _my_value;
  value_type* _local_memory;
  value_type* _local_output;
  value_type* _global_input;
  bool _is_final_stage;
};

}
}
}

#endif