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

#ifndef HIPSYCL_GLUE_HOST_SEQUENTIAL_REDUCER_HPP
#define HIPSYCL_GLUE_HOST_SEQUENTIAL_REDUCER_HPP

#include <new>
#include <vector>

#include "hipSYCL/glue/generic/reduction_accumulator.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/reduction.hpp"

namespace hipsycl {
namespace glue {
namespace host {

#ifndef HIPSYCL_FORCE_CACHE_LINE_SIZE
#define HIPSYCL_FORCE_CACHE_LINE_SIZE 128
#endif

#ifdef HIPSYCL_FORCE_CACHE_LINE_SIZE
constexpr std::size_t cache_line_size = HIPSYCL_FORCE_CACHE_LINE_SIZE;
#else
// This C++17 feature is unfortunately not yet widely supported
constexpr std::size_t cache_line_size =
    std::hardware_destructive_interference_size;
#endif

template <class T> struct cache_line_aligned {
  alignas(cache_line_size) T data;
};

template<class ReductionDescriptor>
class sequential_reducer {
public:
  using value_type = typename ReductionDescriptor::value_type;
  using combiner_type = typename ReductionDescriptor::combiner_type;
  using accumulator_type = sequential_reduction_accumulator<ReductionDescriptor>;

  sequential_reducer(int num_threads, ReductionDescriptor &desc)
      : _desc{desc}, _per_thread_results(num_threads, {accumulator_type{desc}}) {}

  void combine(int my_thread_id, const value_type& v) {
    assert(my_thread_id < _per_thread_results.size());
    _per_thread_results[my_thread_id].data.combine_with(_desc, v);
  }

  // This should be executed in a single threaded scope.
  // Sums up all the partial results and stores in the result data buffer
  void finalize_result() {
    bool initialize_from_dest = true;
    if constexpr (ReductionDescriptor::has_identity) {
      initialize_from_dest = !_desc.initialize_to_identity;
    }
    accumulator_type accumulator(_desc);
    if (initialize_from_dest) {
      accumulator.combine_with(_desc, *_desc.get_pointer());
    }
    for (std::size_t i = 0; i < _per_thread_results.size(); ++i) {
      accumulator.combine_with(_desc, _per_thread_results[i].data);
    }
    *(_desc.get_pointer()) = accumulator.value;
  }

  value_type identity() const {
    return _desc.identity;
  }

private:
  ReductionDescriptor &_desc;
  // TODO: new does not necessarily respect over-aligned alignas requirements.
  // Depending on the value of std::max_align_t and cache_line_size,
  // alignment may be off and not match cache lines.
  std::vector<cache_line_aligned<accumulator_type>> _per_thread_results;
};

}
}
}

#endif
