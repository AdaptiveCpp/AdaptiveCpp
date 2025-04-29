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
#ifndef HIPSYCL_SP_ITEM_HPP
#define HIPSYCL_SP_ITEM_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "id.hpp"
#include "range.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

template<int Dim>
class sp_item
{
  template <int D>
  ACPP_KERNEL_TARGET friend sp_item<D>
  make_sp_item(sycl::id<D> local_id, sycl::id<D> global_id,
               sycl::range<D> local_range,
               sycl::range<D> global_range) noexcept;

public:
  static constexpr int dimensions = Dim;

  ACPP_KERNEL_TARGET
  sycl::range<Dim> get_global_range() const noexcept {
    return _global_range;
  }

  ACPP_KERNEL_TARGET
  size_t get_global_range(int dimension) const noexcept {
    return _global_range[dimension];
  }

  ACPP_KERNEL_TARGET
  size_t get_global_linear_range() const noexcept {
    return _global_range.size();
  }

  ACPP_KERNEL_TARGET
  sycl::id<Dim> get_global_id() const noexcept {
    return _global_id;
  }

  ACPP_KERNEL_TARGET
  size_t get_global_id(int dimension) const noexcept {
    return _global_id[dimension];
  }

  ACPP_KERNEL_TARGET
  size_t get_global_linear_id() const noexcept {
    return detail::linear_id<Dim>::get(get_global_id(), get_global_range());
  }

  ACPP_KERNEL_TARGET
  sycl::range<Dim> get_innermost_local_range() const noexcept {
    return _local_range;
  }

  ACPP_KERNEL_TARGET
  size_t get_innermost_local_range(int dimension) const noexcept {
    return _local_range[dimension];
  }

  ACPP_KERNEL_TARGET
  size_t get_innermost_local_linear_range() const noexcept {
    return _local_range.size();
  }

  ACPP_KERNEL_TARGET
  sycl::id<Dim> get_innermost_local_id() const noexcept {
    return _local_id;
  }

  ACPP_KERNEL_TARGET
  size_t get_innermost_local_id(int dimensions) const noexcept {
    return _local_id[dimensions];
  }

  ACPP_KERNEL_TARGET
  size_t get_innermost_local_linear_id() const noexcept {
    return detail::linear_id<Dim>::get(get_innermost_local_id(),
                                       get_innermost_local_range());
  }

  template<class Group>
  ACPP_KERNEL_TARGET
  sycl::id<Dim> get_local_id(const Group& g) const noexcept {
    return g.get_local_id(*this);
  }

  template<class Group>
  ACPP_KERNEL_TARGET
  size_t get_local_id(const Group& g, int dimension) const noexcept {
    return g.get_local_id(*this, dimension);
  }

  template<class Group>
  ACPP_KERNEL_TARGET
  size_t get_local_linear_id(const Group& g) const noexcept {
    return g.get_local_linear_id(*this);
  }


  template<class Group>
  ACPP_KERNEL_TARGET
  sycl::range<Dim> get_local_range(const Group& grp) const noexcept {
    return grp.get_logical_local_range();
  }

  template<class Group>
  ACPP_KERNEL_TARGET
  size_t get_local_range(const Group& grp, int dimension) const noexcept {
    return grp.get_logical_local_range(dimension);
  }

  template<class Group>
  ACPP_KERNEL_TARGET
  size_t get_local_linear_range(const Group& grp) const noexcept {
    return grp.get_logical_local_linear_range();
  }

private:
  sp_item(sycl::id<Dim> local_id, sycl::id<Dim> global_id,
          sycl::range<Dim> local_range, sycl::range<Dim> global_range) noexcept
      : _local_id{local_id}, _global_id{global_id}, _local_range{local_range},
        _global_range{global_range} {}

  const sycl::id<Dim> _local_id;
  const sycl::id<Dim> _global_id;

  const sycl::range<Dim> _local_range;
  const sycl::range<Dim> _global_range;
};

template <int Dim>
ACPP_KERNEL_TARGET sp_item<Dim>
make_sp_item(sycl::id<Dim> local_id, sycl::id<Dim> global_id,
             sycl::range<Dim> local_range,
             sycl::range<Dim> global_range) noexcept {

  return sp_item<Dim>{local_id, global_id, local_range, global_range};
}
} // namespace detail

// deprecated
template<int Dim>
using logical_item=detail::sp_item<Dim>;

// deprecated
template<int Dim>
using physical_item=detail::sp_item<Dim>;

template<int Dim>
using s_item=detail::sp_item<Dim>;

}
}

#endif
