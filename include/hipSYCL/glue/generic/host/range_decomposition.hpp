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
#ifndef HIPSYCL_RANGE_DECOMPOSITION_HPP
#define HIPSYCL_RANGE_DECOMPOSITION_HPP

#include <vector>
#include <cassert>

#include "hipSYCL/sycl/libkernel/id.hpp"
#include "hipSYCL/sycl/libkernel/range.hpp"

#include "iterate_range.hpp"

namespace hipsycl {
namespace glue {
namespace host {

template<int Dim>
class static_range_decomposition {
public:
  static_assert(Dim >= 1 && Dim <= 3, "Dimension must be 1,2 or 3");

  static_range_decomposition(sycl::range<Dim> r, int num_regions)
      : _range{r}, _regions_begin(num_regions),
        _regions_size(num_regions) {

    std::size_t total_num_elements = _range.size();

    std::size_t remainder = total_num_elements % num_regions;
    std::size_t region_size = total_num_elements / num_regions;

    for (std::size_t i = 0; i < num_regions; ++i) {
      _regions_size[i] = region_size;
      if (i < remainder)
        ++_regions_size[i];
    }

    std::size_t begin = 0;
    for (std::size_t i = 0; i < num_regions; ++i) {

      sycl::id<Dim> nd_begin;

      if constexpr (Dim == 1) {
        nd_begin[0] = begin;
      } else if constexpr (Dim == 2) {
      
        nd_begin[1] = begin % _range[1];
        nd_begin[0] = begin / _range[1];
      
      } else if constexpr (Dim == 3) {
        std::size_t surface_id = begin / (_range[2] * _range[1]);
        std::size_t index2d    = begin % (_range[2] * _range[1]);
        
        nd_begin[2] = index2d % _range[2];
        nd_begin[1] = index2d / _range[2];
        nd_begin[0] = surface_id;
      }

      _regions_begin[i] = nd_begin;
      begin += _regions_size[i];
    }
    assert(begin == total_num_elements);
  }

  template <class F> void for_each_local_element(int region_id, F f) const {
    assert(region_id < _regions_begin.size());

    iterate_partial_range(_range, _regions_begin[region_id],
                          _regions_size[region_id], f);
  }

private:
  sycl::range<Dim> _range;
  std::vector<sycl::id<Dim>> _regions_begin;
  std::vector<std::size_t> _regions_size;
};

}
}
}

#endif
