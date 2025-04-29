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
#ifndef HIPSYCL_ITEM_HPP
#define HIPSYCL_ITEM_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "id.hpp"
#include "range.hpp"
#include "detail/data_layout.hpp"

#include <type_traits>

namespace hipsycl {
namespace sycl {

template <int Dimensions, bool with_offset>
struct item;

namespace detail {

template <int Dimensions>
struct item_base
{
protected:
  struct not_convertible_to_scalar {};

  static constexpr auto get_scalar_conversion_type() {
    if constexpr(Dimensions == 1)
      return std::size_t{};
    else
      return not_convertible_to_scalar {};
  }

  using scalar_conversion_type = decltype(get_scalar_conversion_type());
public:
  static constexpr int dimensions = Dimensions;

  ACPP_KERNEL_TARGET 
  item_base(const sycl::id<Dimensions>& my_id,
    const sycl::range<Dimensions>& global_size)
    : global_id{my_id}, global_size(global_size)
  {}

  /* -- common interface members -- */

  ACPP_KERNEL_TARGET 
  sycl::range<Dimensions> get_range() const
  { return global_size; }

  ACPP_KERNEL_TARGET 
  size_t get_range(int dimension) const
  { return global_size[dimension]; }

  ACPP_KERNEL_TARGET sycl::id<Dimensions> get_id() const
  {
    return this->global_id;
  }

  ACPP_KERNEL_TARGET size_t get_id(int dimension) const
  {
    return this->global_id[dimension];
  }

  ACPP_KERNEL_TARGET size_t operator[](int dimension) const
  {
    return this->global_id[dimension];
  }

  ACPP_KERNEL_TARGET size_t get_linear_id() const
  {
    return detail::linear_id<Dimensions>::get(this->global_id,
      this->global_size);
  }

protected:
  sycl::id<Dimensions> global_id;
  sycl::range<Dimensions> global_size;
};

/// Creates an Item with offset.
/// \param my_effective_id This has to be global_id + offset.
template <int Dimensions>
ACPP_KERNEL_TARGET
item<Dimensions, true> make_item(const sycl::id<Dimensions>& my_effective_id,
  const sycl::range<Dimensions>& global_size, const sycl::id<Dimensions>& offset)
{
  return item<Dimensions, true>{my_effective_id, global_size, offset};
}

/// Creates an Item without offset
/// \param my_id This should equal global_id
template <int Dimensions>
ACPP_KERNEL_TARGET
item<Dimensions, false> make_item(const sycl::id<Dimensions>& my_id,
  const sycl::range<Dimensions>& global_size)
{
  return item<Dimensions, false>{my_id, global_size};
}

} // namespace detail

template <int Dimensions = 1, bool with_offset = true>
struct item : detail::item_base<Dimensions>
{};

template <int Dimensions>
struct item<Dimensions, true> : detail::item_base<Dimensions>
{
  ACPP_KERNEL_TARGET sycl::id<Dimensions> get_offset() const
  {
    return offset;
  }

  ACPP_KERNEL_TARGET friend bool operator ==(const item<Dimensions, true> lhs, const item<Dimensions, true> rhs)
  {
    return lhs.global_id == rhs.global_id &&
           lhs.global_size == rhs.global_size &&
           lhs.offset == rhs.offset;
  }

  ACPP_KERNEL_TARGET friend bool operator !=(const item<Dimensions, true> lhs, const item<Dimensions, true> rhs)
  {
    return !(lhs==rhs);
  }

  // We cannot use enable_if since the involved templates would
  // prevent implicit type conversion to other integer types.
  ACPP_UNIVERSAL_TARGET
  operator typename detail::item_base<Dimensions>::scalar_conversion_type()
      const {
    return this->global_id[0];
  }

private:
  template<int d>
  using _range = sycl::range<d>; // workaround for nvcc

  friend ACPP_KERNEL_TARGET 
  item<Dimensions, true> detail::make_item<Dimensions>(
    const sycl::id<Dimensions>&, const _range<Dimensions>&, 
    const sycl::id<Dimensions>&);

  ACPP_KERNEL_TARGET 
  item(const sycl::id<Dimensions>& my_id,
    const sycl::range<Dimensions>& global_size, 
    const sycl::id<Dimensions>& offset)
    : detail::item_base<Dimensions>(my_id, global_size), offset{offset}
  {}

  sycl::id<Dimensions> offset;
};

template <int Dimensions>
struct item<Dimensions, false> : detail::item_base<Dimensions>
{
  ACPP_KERNEL_TARGET operator item<Dimensions, true>() const
  {
    return detail::make_item<Dimensions>(this->global_id, this->global_size,
      sycl::id<Dimensions>{});
  }

  ACPP_KERNEL_TARGET friend bool operator ==(const item<Dimensions, false> lhs, const item<Dimensions, false> rhs)
  {
    return lhs.global_id == rhs.global_id &&
           lhs.global_size == rhs.global_size;
  }

  ACPP_KERNEL_TARGET friend bool operator !=(const item<Dimensions, false> lhs, const item<Dimensions, false> rhs)
  {
    return !(lhs==rhs);
  }

  // We cannot use enable_if since the involved templates would
  // prevent implicit type conversion to other integer types.
  ACPP_UNIVERSAL_TARGET
  operator typename detail::item_base<Dimensions>::scalar_conversion_type()
      const {
    return this->global_id[0];
  }
private:
  template<int d>
  using _range = sycl::range<d>; // workaround for nvcc
  friend ACPP_KERNEL_TARGET item<Dimensions, false> detail::make_item<Dimensions>(
    const sycl::id<Dimensions>&, const _range<Dimensions>&);

  ACPP_KERNEL_TARGET item(const sycl::id<Dimensions>& my_id,
    const sycl::range<Dimensions>& global_size)
    : detail::item_base<Dimensions>(my_id, global_size)
  {}
};

}
}

#endif
