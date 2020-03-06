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

#include "../id.hpp"
#include "../range.hpp"
#include "../backend/backend.hpp"
#include "../types.hpp"
#include "../exception.hpp"

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

template<int dim>
class data_layout
{
public:
  static_assert(dim >= 0 && dim <= 3,
                "Number of dimensions must be "
                "between 0 and 3");

  data_layout(const sycl::range<dim>& accessed_range,
              const sycl::id<dim>& access_offset,
              const sycl::range<dim>& data_shape)
    : _access_range{accessed_range},
      _offset{access_offset},
      _shape{data_shape}
  {}

  template<class Function>
  void for_each_contiguous_memory_region(Function f) const
  {
    if(dim == 0)
    {
      f(linear_data_range{0, 1});
    }
    else if(dim == 1)
    {
      f(linear_data_range{_offset.get(0),_access_range.get(0)});
    }
    else if(dim == 2)
    {
      for_each_contiguous_region2D(f);
    }
    else if(dim == 3)
    {
      for_each_contiguous_region3D(f);
    }
    else
    {
      // Should never happen because of our static_assert();
      assert(false);
    }
  }

private:
  template<class Function>
  void for_each_contiguous_region2D(Function f) const
  {
    if(_access_range.get(1) == _shape.get(1))
    {
      assert(_offset.get(1) == 0);
      f(linear_data_range{linear_id<2>::get(_offset,_shape),
                          _access_range.size()});
    }
    else
    {
      // Shapes of fastest index do not match, we must
      // iterate over all contiguous 1D regions
      size_t end = _offset.get(0) + _access_range.get(0);
      for(size_t x = _offset.get(0);
          x < end;
          ++x)
      {
        f(linear_data_range{linear_id<2>::get(sycl::id<2>{x,_offset.get(1)},_shape),
                            _access_range.get(1)});
      }
    }
  }


  template<class Function>
  void for_each_contiguous_region3D(Function f) const
  {
    if(_access_range.get(1) == _shape.get(1))
    {
      assert(_offset.get(1) == 0);
      if(_access_range.get(2) == _shape.get(2))
      {
        assert(_offset.get(2) == 0);

        // Shapes of the fastest two indices match,
        // so we are accessing full, contiguous 2d slices of the
        // 3d range
        f(linear_data_range{linear_id<3>::get(_offset,_shape),
                            _access_range.size()});
      }
      else
      {
        // Only shape of the fastest index matches,
        // we are at least accessing partial 2D slices
        // and must iterate over all 2D slices
        size_t end_x = _offset.get(0) + _access_range.get(0);
        for(size_t x = _offset.get(0);
            x < end_x;
            ++x)
        {
          f(linear_data_range{
              linear_id<3>::get(sycl::id<3>{x,_offset.get(1),_offset.get(2)},_shape),
              _access_range.get(1) * _access_range.get(2)
            });
        }

      }
    }
    else
    {
      // We must iterate over all contiguous 1D ranges
      size_t end_x = _offset.get(0) + _access_range.get(0);
      size_t end_y = _offset.get(1) + _access_range.get(1);
      for(size_t x = _offset.get(0);
          x < end_x;
          ++x)
      {
        for(size_t y = _offset.get(1);
            y < end_y;
            ++y)
        {
          f(linear_data_range{
              linear_id<3>::get(sycl::id<3>{x,y,_offset.get(2)},_shape),
              _access_range.get(2)
            });
        }
      }
    }
  }

  sycl::range<dim> _access_range;
  sycl::id<dim> _offset;
  sycl::range<dim> _shape;
};

}
}
}

#endif
