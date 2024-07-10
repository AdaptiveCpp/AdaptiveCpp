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
#include <cstddef>
#include <cstdint>

#ifndef HIPSYCL_WG_CONFIGURED_REDUCTION_DESCRIPTOR_HPP
#define HIPSYCL_WG_CONFIGURED_REDUCTION_DESCRIPTOR_HPP

#include "../reduction_descriptor.hpp"

namespace hipsycl::algorithms::reduction::wg_model {

/// In addition to reduction_descriptor, also stores internal information
/// needed by the reduction engine, such as scratch data pointers.
/// This object will be constructed by the reduction engine on the host,
/// but its member functions should only be used on device!
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
        _problem_size{problem_size} {}

  bool is_final_stage() const noexcept {
    return !_stage_output;
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
    if (!_stage_output)
      return this->get_final_output_destination();
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
