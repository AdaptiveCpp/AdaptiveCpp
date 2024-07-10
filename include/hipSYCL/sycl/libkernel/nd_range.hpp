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
#ifndef HIPSYCL_ND_RANGE_HPP
#define HIPSYCL_ND_RANGE_HPP

#include "range.hpp"
#include "id.hpp"

namespace hipsycl {
namespace sycl {

template<int Dimensions = 1>
struct nd_range {
  static constexpr int dimensions = Dimensions;

  ACPP_UNIVERSAL_TARGET
  nd_range(range<Dimensions> globalSize,
           range<Dimensions> localSize,
           id<Dimensions> offset = id<Dimensions>())
    : _global_range{globalSize},
      _local_range{localSize},
      _num_groups{globalSize / localSize},
      _offset{offset}
  {}

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_global() const
  { return _global_range; }

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_global_range() const
  { return get_global(); }

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_local() const
  { return _local_range; }

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_local_range() const
  { return get_local(); }

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_group() const
  { return _num_groups; }

  ACPP_UNIVERSAL_TARGET
  range<Dimensions> get_group_range() const
  { return get_group(); }

  ACPP_UNIVERSAL_TARGET
  id<Dimensions> get_offset() const
  { return _offset; }
  
  friend bool operator==(const nd_range<Dimensions>& lhs, const nd_range<Dimensions>& rhs)
  {
    return lhs._global_range == rhs._global_range &&
           lhs._local_range == rhs._local_range &&
           lhs._num_groups == rhs._num_groups &&
           lhs._offset == rhs._offset;
  }

  friend bool operator!=(const nd_range<Dimensions>& lhs, const nd_range<Dimensions>& rhs){
    return !(lhs == rhs);
  }

private:
  range<Dimensions> _global_range;
  range<Dimensions> _local_range;
  range<Dimensions> _num_groups;
  id<Dimensions> _offset;
};


}
}

#endif
