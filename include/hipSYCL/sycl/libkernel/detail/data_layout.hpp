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

#ifndef HIPSYCL_DATA_LAYOUT_HPP
#define HIPSYCL_DATA_LAYOUT_HPP

#include <cassert>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/types.hpp"
#include "../id.hpp"
#include "../range.hpp"


namespace hipsycl {
namespace sycl {
namespace detail {


inline HIPSYCL_UNIVERSAL_TARGET size_t get_linear_id(const size_t id_x,
                                                const size_t id_y,
                                                const size_t range_y)
{
  return id_x * range_y + id_y;
}

inline HIPSYCL_UNIVERSAL_TARGET size_t get_linear_id(const size_t id_x,
                                                const size_t id_y,
                                                const size_t id_z,
                                                const size_t range_y,
                                                const size_t range_z)
{
  return id_x * range_y * range_z + id_y * range_z + id_z;
}

template<int dim>
struct linear_id
{
};

template<>
struct linear_id<1>
{
  static HIPSYCL_UNIVERSAL_TARGET size_t get(const sycl::id<1>& idx)
  { return idx[0]; }

  static HIPSYCL_UNIVERSAL_TARGET size_t get(const sycl::id<1>& idx,
                                            const sycl::range<1>& r)
  {
    return get(idx);
  }
};

template<>
struct linear_id<2>
{
  static HIPSYCL_UNIVERSAL_TARGET size_t get(const sycl::id<2>& idx,
                                        const sycl::range<2>& r)
  {
    return get_linear_id(idx.get(0), idx.get(1), r.get(1));
  }
};

template<>
struct linear_id<3>
{
  static HIPSYCL_UNIVERSAL_TARGET size_t get(const sycl::id<3>& idx,
                                        const sycl::range<3>& r)
  {
    return get_linear_id(idx.get(0), idx.get(1), idx.get(2), r.get(1), r.get(2));
  }
};

struct linear_data_range
{
  size_t begin;
  size_t num_elements;
};


}
}
}

#endif
