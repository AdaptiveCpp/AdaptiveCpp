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
#ifndef HIPSYCL_PRIVATE_MEMORY_HPP
#define HIPSYCL_PRIVATE_MEMORY_HPP

#include <memory>

#include "group.hpp"
#include "h_item.hpp"

namespace hipsycl {
namespace sycl {

#ifdef SYCL_DEVICE_ONLY

template<typename T, int Dimensions = 1>
class private_memory
{
public:
  ACPP_KERNEL_TARGET
  private_memory(const group<Dimensions>&)
  {}

  ACPP_KERNEL_TARGET
  T& operator()(const h_item<Dimensions>&) noexcept
  {
    return _data;
  }

private:
  T _data;
};

#else

template<typename T, int Dimensions = 1>
class private_memory
{
public:
  ACPP_KERNEL_TARGET
  private_memory(const group<Dimensions>& grp)
  : _data{new T [grp.get_local_range().size()]}
  {}

  ACPP_KERNEL_TARGET
  T& operator()(const h_item<Dimensions>& idx) noexcept
  {
    return get(idx.get_local_id(), idx.get_local_range());
  }

private:
  std::unique_ptr<T []> _data;

  ACPP_KERNEL_TARGET
  T& get(id<Dimensions> id, range<Dimensions> local_range) noexcept
  {
    return _data.get()[detail::linear_id<Dimensions>::get(id, local_range)];
  }
};
#endif

}
}

#endif
