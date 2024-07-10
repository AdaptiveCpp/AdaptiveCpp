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


inline ACPP_UNIVERSAL_TARGET size_t get_linear_id(const size_t id_x,
                                                  const size_t id_y,
                                                  const size_t range_y)
{
  return id_x * range_y + id_y;
}

inline ACPP_UNIVERSAL_TARGET size_t get_linear_id(const size_t id_x,
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
  static ACPP_UNIVERSAL_TARGET size_t get(const sycl::id<1>& idx)
  { return idx[0]; }

  static ACPP_UNIVERSAL_TARGET size_t get(const sycl::id<1>& idx,
                                          const sycl::range<1>& r)
  {
    return get(idx);
  }
};

template<>
struct linear_id<2>
{
  static ACPP_UNIVERSAL_TARGET size_t get(const sycl::id<2>& idx,
                                          const sycl::range<2>& r)
  {
    return get_linear_id(idx.get(0), idx.get(1), r.get(1));
  }
};

template<>
struct linear_id<3>
{
  static ACPP_UNIVERSAL_TARGET size_t get(const sycl::id<3>& idx,
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
