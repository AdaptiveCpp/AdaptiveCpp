/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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


#ifndef HIPSYCL_ND_RANGE_HPP
#define HIPSYCL_ND_RANGE_HPP

#include "range.hpp"
#include "id.hpp"

namespace hipsycl {
namespace sycl {

template<int Dimensions = 1>
struct nd_range {
  static constexpr int dimensions = Dimensions;

  HIPSYCL_UNIVERSAL_TARGET
  nd_range(range<Dimensions> globalSize,
           range<Dimensions> localSize,
           id<Dimensions> offset = id<Dimensions>())
    : _global_range{globalSize},
      _local_range{localSize},
      _num_groups{globalSize / localSize},
      _offset{offset}
  {}

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_global() const
  { return _global_range; }

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_global_range() const
  { return get_global(); }

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_local() const
  { return _local_range; }

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_local_range() const
  { return get_local(); }

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_group() const
  { return _num_groups; }

  HIPSYCL_UNIVERSAL_TARGET
  range<Dimensions> get_group_range() const
  { return get_group(); }

  HIPSYCL_UNIVERSAL_TARGET
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
