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

#ifndef HIPSYCL_H_ITEM_HPP
#define HIPSYCL_H_ITEM_HPP

#include "item.hpp"

namespace cl {
namespace sycl {

template<int dimensions>
struct group;

template <int dimensions>
struct h_item
{
  friend struct group<dimensions>;

  HIPSYCL_KERNEL_TARGET
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
  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_global() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::make_item<dimensions>(
      detail::get_global_id<dimensions>(),
      detail::get_global_size<dimensions>()
    );
#else
    assert(false && "Host execution when compiling for CUDA/HIP is unsupported");
    return detail::invalid_host_call_dummy_return(
      detail::make_item<dimensions>(
        id<dimensions>{},
        range<dimensions>{}));
    
#endif
  }

  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_local() const
  {
    return get_logical_local();
  }

  /// \return The local id in the logical iteration space.
  ///
  /// \todo This currently always returns the physical local id
  /// since flexible work group ranges are currently unsupported.
  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_logical_local() const
  {
    return get_physical_local();
  }

  HIPSYCL_KERNEL_TARGET
  item<dimensions, false> get_physical_local() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::make_item<dimensions>(
      detail::get_local_id<dimensions>(),
      detail::get_global_size<dimensions>()
    );
#else
    return detail::invalid_host_call_dummy_return(
      detail::make_item<dimensions>(
        id<dimensions>{},
        range<dimensions>{}));
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_global_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_global_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_id<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<id<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_global_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_global_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_local_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_local_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<id<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_local_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  /// \todo This always returns the physical range.
  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_logical_local_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  /// \todo This always returns the physical range.
  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  /// \todo This always returns the physical id
  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_logical_local_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<id<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  range<dimensions> get_physical_local_range() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<range<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_size(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  id<dimensions> get_physical_local_id() const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id<dimensions>();
#else
    return detail::invalid_host_call_dummy_return<id<dimensions>>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const
  {
#ifdef __HIPSYCL_DEVICE_CALLABLE__
    return detail::get_local_id(dimension);
#else
    return detail::invalid_host_call_dummy_return<size_t>();
#endif
  }
};

}
}

#endif
