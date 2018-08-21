/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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

#ifndef SYCU_H_ITEM_HPP
#define SYCU_H_ITEM_HPP

#include "item.hpp"

namespace cl {
namespace sycl {

template<int dimensions>
class group;

template <int dimensions>
struct h_item
{
  friend class group<dimensions>;

  __device__
  h_item(){}
public:
  /* -- common interface members -- */

  /// \return The global id with respect to the parallel_for_work_group
  /// invocation. Flexlible local ranges are not taken into account.
  ///
  /// \todo Since parallel_for_work_group is processed per group,
  /// does this imply that we have one "global id" per group? Or should
  /// it take into account the number of threads that the implementation
  /// internally spawns? For the moment, we use the latter interpretation.
  __device__
  item<dimensions, false> get_global() const
  {
    return item<dimensions, false>{};
  }

  __device__
  item<dimensions, false> get_local() const
  {
    return get_logical_local();
  }

  /// \return The local id in the logical iteration space.
  ///
  /// \todo This currently always returns the physical local id
  /// since flexible work group ranges are currently unsupported.
  __device__
  item<dimensions, false> get_logical_local() const
  {
    return get_physical_local();
  }

  __device__
  item<dimensions, false> get_physical_local() const
  {
    return item<dimensions, false>{detail::item_impl<dimensions>{
        detail::get_local_id<dimensions>()
      }
    };
  }

  __device__
  range<dimensions> get_global_range() const
  {
    return detail::get_global_size<dimensions>();
  }

  __device__
  size_t get_global_range(int dimension) const
  {
    return detail::get_global_size(dimension);
  }

  __device__
  id<dimensions> get_global_id() const
  {
    return detail::get_global_id<dimensions>();
  }

  __device__
  size_t get_global_id(int dimension) const
  {
    return detail::get_global_id(dimension);
  }

  __device__
  range<dimensions> get_local_range() const
  {
    return detail::get_local_size<dimensions>();
  }

  __device__
  size_t get_local_range(int dimension) const
  {
    return detail::get_local_size(dimension);
  }

  __device__
  id<dimensions> get_local_id() const
  {
    return detail::get_local_id<dimensions>();
  }

  __device__
  size_t get_local_id(int dimension) const
  {
    return detail::get_local_id(dimension);
  }

  /// \todo This always returns the physical range.
  __device__
  range<dimensions> get_logical_local_range() const
  {
    return detail::get_local_size<dimensions>();
  }

  /// \todo This always returns the physical range.
  __device__
  size_t get_logical_local_range(int dimension) const
  {
    return detail::get_local_size(dimension);
  }

  /// \todo This always returns the physical id
  __device__
  id<dimensions> get_logical_local_id() const
  {
    return detail::get_local_id<dimensions>();
  }

  __device__
  size_t get_logical_local_id(int dimension) const
  {
    return detail::get_local_id(dimension);
  }

  __device__
  range<dimensions> get_physical_local_range() const
  {
    return detail::get_local_size<dimensions>();
  }

  __device__
  size_t get_physical_local_range(int dimension) const
  {
    return detail::get_local_size(dimension);
  }

  __device__
  id<dimensions> get_physical_local_id() const
  {
    return detail::get_local_id<dimensions>();
  }

  __device__
  size_t get_physical_local_id(int dimension) const
  {
    return detail::get_local_id(dimension);
  }
};

}
}

#endif
