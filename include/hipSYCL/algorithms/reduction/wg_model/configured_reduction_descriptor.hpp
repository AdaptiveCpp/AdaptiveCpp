
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

#include <cstddef>
#include <cstdint>

#ifndef HIPSYCL_WG_CONFIGURED_REDUCTION_DESCRIPTOR_HPP
#define HIPSYCL_WG_CONFIGURED_REDUCTION_DESCRIPTOR_HPP

#include "../reduction_descriptor.hpp"

namespace hipsycl::algorithms::reduction::wg_model {

/// In addition to reduction_descriptor, also stores internal information
/// needed by the reduction engine, such as scratch data pointers.
/// This object will be constructed by the reduction engine.
template <class ReductionDescriptor>
class configured_reduction_descriptor : public ReductionDescriptor {
public:
  // intermediate stage reduction
  configured_reduction_descriptor(
      // The basic reduction descriptor
      const ReductionDescriptor &basic_descriptor,
      // Array describing whether the input was initialized. Irrelevant if the
      // identity is known, or we are in the first reduction stage
      initialization_flag_t *is_input_initialized,
      // Array where information will be stored whether the output was
      // initialized. Irrelevant for the last stage or if the identity is known.
      initialization_flag_t *is_output_initialized,
      // Input for the stage. Irrelevant for the first stage.
      typename ReductionDescriptor::value_type *stage_input,
      // Output of the stage. If nullptr, assumes final stage & overall
      // reduction output
      typename ReductionDescriptor::value_type *stage_output,
      std::size_t problem_size)
      : ReductionDescriptor{basic_descriptor},
        _is_input_initialized{is_input_initialized},
        _is_output_initialized{is_output_initialized},
        _stage_input{stage_input}, _stage_output{stage_output},
        _problem_size{problem_size} {
    if (!_stage_output)
      _stage_output = this->get_final_output_destination();
  }

  bool is_final_stage() const noexcept {
    return _stage_output == this->get_final_output_destination();
  }

  initialization_flag_t *get_input_initialization_state() const noexcept {
    return _is_input_initialized;
  }

  initialization_flag_t *get_output_initialization_state() const noexcept {
    return _is_output_initialized;
  }

  typename ReductionDescriptor::value_type *get_stage_input() const noexcept {
    return _stage_input;
  }

  typename ReductionDescriptor::value_type *get_stage_output() const noexcept {
    return _stage_output;
  }

  std::size_t get_problem_size() const noexcept { return _problem_size; }

private:
  initialization_flag_t *_is_input_initialized;
  initialization_flag_t *_is_output_initialized;
  typename ReductionDescriptor::value_type *_stage_input;
  typename ReductionDescriptor::value_type *_stage_output;
  std::size_t _problem_size;
};

}


#endif
