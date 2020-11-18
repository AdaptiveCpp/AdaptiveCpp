/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_ITEM_HPP
#define HIPSYCL_ITEM_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "id.hpp"
#include "range.hpp"
#include "detail/data_layout.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {

template <int dimensions, bool with_offset>
struct item;

namespace detail {

template <int dimensions>
struct item_base
{
  HIPSYCL_KERNEL_TARGET 
  item_base(const sycl::id<dimensions>& my_id,
    const sycl::range<dimensions>& global_size)
    : global_id{my_id}, global_size(global_size)
  {}

  /* -- common interface members -- */

  HIPSYCL_KERNEL_TARGET 
  sycl::range<dimensions> get_range() const
  { return global_size; }

  HIPSYCL_KERNEL_TARGET 
  size_t get_range(int dimension) const
  { return global_size[dimension]; }

  HIPSYCL_KERNEL_TARGET sycl::id<dimensions> get_id() const
  {
    return this->global_id;
  }

  HIPSYCL_KERNEL_TARGET size_t get_id(int dimension) const
  {
    return this->global_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET size_t operator[](int dimension) const
  {
    return this->global_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET size_t get_linear_id() const
  {
    return detail::linear_id<dimensions>::get(this->global_id,
      this->global_size);
  }
protected:
  sycl::id<dimensions> global_id;
  sycl::range<dimensions> global_size;
};

/// Creates an Item with offset.
/// \param my_effective_id This has to be global_id + offset.
template <int dimensions>
HIPSYCL_KERNEL_TARGET
item<dimensions, true> make_item(const sycl::id<dimensions>& my_effective_id,
  const sycl::range<dimensions>& global_size, const sycl::id<dimensions>& offset)
{
  return item<dimensions, true>{my_effective_id, global_size, offset};
}

/// Creates an Item without offset
/// \param my_id This should equal global_id
template <int dimensions>
HIPSYCL_KERNEL_TARGET
item<dimensions, false> make_item(const sycl::id<dimensions>& my_id,
  const sycl::range<dimensions>& global_size)
{
  return item<dimensions, false>{my_id, global_size};
}

} // namespace detail

template <int dimensions = 1, bool with_offset = true>
struct item : detail::item_base<dimensions>
{};

template <int dimensions>
struct item<dimensions, true> : detail::item_base<dimensions>
{
  HIPSYCL_KERNEL_TARGET sycl::id<dimensions> get_offset() const
  {
    return offset;
  }

  HIPSYCL_KERNEL_TARGET friend bool operator ==(const item<dimensions, true> lhs, const item<dimensions, true> rhs)
  {
    return lhs.my_id == rhs.my_id &&
           lhs.global_id == rhs.global_id &&
           lhs.global_size == rhs.global_size &&
           lhs.offset == rhs.offset;
  }

  HIPSYCL_KERNEL_TARGET friend bool operator !=(const item<dimensions, true> lhs, const item<dimensions, true> rhs)
  {
    return !(lhs==rhs);
  }

private:
  template<int d>
  using _range = sycl::range<d>; // workaround for nvcc

  friend HIPSYCL_KERNEL_TARGET 
  item<dimensions, true> detail::make_item<dimensions>(
    const sycl::id<dimensions>&, const _range<dimensions>&, 
    const sycl::id<dimensions>&);

  HIPSYCL_KERNEL_TARGET 
  item(const sycl::id<dimensions>& my_id,
    const sycl::range<dimensions>& global_size, 
    const sycl::id<dimensions>& offset)
    : detail::item_base<dimensions>(my_id, global_size), offset{offset}
  {}

  const sycl::id<dimensions> offset;
};

template <int dimensions>
struct item<dimensions, false> : detail::item_base<dimensions>
{
  HIPSYCL_KERNEL_TARGET operator item<dimensions, true>() const
  {
    return detail::make_item<dimensions>(this->global_id, this->global_size,
      sycl::id<dimensions>{});
  }

  HIPSYCL_KERNEL_TARGET friend bool operator ==(const item<dimensions, false> lhs, const item<dimensions, false> rhs)
  {
    return lhs.my_id == rhs.my_id &&
           lhs.global_id == rhs.global_id &&
           lhs.global_size == rhs.global_size;
  }

  HIPSYCL_KERNEL_TARGET friend bool operator !=(const item<dimensions, false> lhs, const item<dimensions, false> rhs)
  {
    return !(lhs==rhs);
  }

private:
  template<int d>
  using _range = sycl::range<d>; // workaround for nvcc
  friend HIPSYCL_KERNEL_TARGET item<dimensions, false> detail::make_item<dimensions>(
    const sycl::id<dimensions>&, const _range<dimensions>&);

  HIPSYCL_KERNEL_TARGET item(const sycl::id<dimensions>& my_id,
    const sycl::range<dimensions>& global_size)
    : detail::item_base<dimensions>(my_id, global_size)
  {}
};

}
}

#endif
