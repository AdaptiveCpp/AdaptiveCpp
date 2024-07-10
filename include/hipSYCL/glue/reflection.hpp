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
#ifndef HIPSYCL_REFLECTION_HPP
#define HIPSYCL_REFLECTION_HPP

template <class StructT>
void __acpp_introspect_flattened_struct(void *s, int **num_flattened_members,
                                        int **member_offsets,
                                        int **member_sizes, int **member_kinds);

namespace hipsycl::glue::reflection {

enum class type_kind : int { other = 0, pointer = 1, integer_type = 2, float_type = 3 };

class introspect_flattened_struct {
public:
  template<class StructT>
  introspect_flattened_struct(const StructT& s) 
  : _num_members{nullptr} {
    // Compiler needs this copy to figure out
    // the struct type in the presence of opaque pointers.
    // Currently it checks if the first operand of the call
    // to the builtin comes from an alloca instruction.
    StructT s_copy = s;
    __acpp_introspect_flattened_struct<StructT>(&s_copy, &_num_members,
                                                   &_member_offsets, &_member_sizes,
                                                   reinterpret_cast<int **>(&_member_kinds));
  }

  int get_num_members() const {
    if (!_num_members)
      return 0;
    return *_num_members;
  }

  int get_member_offset(int idx) const {
    return _member_offsets[idx];
  }

  int get_member_size(int idx) const {
    return _member_sizes[idx];
  }

  type_kind get_member_kind(int idx) const {
    return _member_kinds[idx];
  }
private:
  int* _num_members;
  int* _member_offsets;
  int* _member_sizes;
  type_kind* _member_kinds;
};

}

#endif
