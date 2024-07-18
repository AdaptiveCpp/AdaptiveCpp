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
#ifndef HIPSYCL_SP_PRIVATE_MEMORY_HPP
#define HIPSYCL_SP_PRIVATE_MEMORY_HPP

#include <memory>

#include "sp_item.hpp"
#include "sp_group.hpp"

namespace hipsycl {
namespace sycl {

namespace detail {

template<typename T, class SpGroup>
class private_memory_access
{
  static constexpr int dimensions = SpGroup::dimensions;
public:
  template <class SG = SpGroup,
            std::enable_if_t<detail::is_sp_group_v<std::decay_t<SG>>, int> = 0>
  ACPP_KERNEL_TARGET explicit private_memory_access(const SpGroup &grp,
                                                       T *data)
      : _data{data}, _grp{grp} {}

  ACPP_KERNEL_TARGET
  T& operator()(const detail::sp_item<dimensions>& idx) const noexcept
  {
    return get(idx.get_local_id(_grp), idx.get_local_range(_grp));
  }

private:
  const SpGroup& _grp;
  T* _data;

  ACPP_KERNEL_TARGET
  T &get(const sycl::id<dimensions> &id,
         const sycl::range<dimensions> &local_range) const noexcept {
    __acpp_if_target_host(
      return _data[detail::linear_id<dimensions>::get(id, local_range)];
    );
    __acpp_if_target_device(
      return *_data;
    );
  }
};

}

#ifdef SYCL_DEVICE_ONLY

template<typename T, class SpGroup>
class s_private_memory
{
  static constexpr int dimensions = SpGroup::dimensions;
public:
  template <class SG = SpGroup,
            std::enable_if_t<detail::is_sp_group_v<SG>, int> = 0>
  [[deprecated("Use sycl::memory_environment() instead")]]
  ACPP_KERNEL_TARGET explicit s_private_memory(const SpGroup & grp)
  {}

  s_private_memory(const s_private_memory&) = delete;
  s_private_memory& operator=(const s_private_memory&) = delete;

  [[deprecated("Use sycl::memory_environment() instead")]]
  ACPP_KERNEL_TARGET
  T& operator()(const detail::sp_item<dimensions>&) noexcept
  {
    return _data;
  }

private:
  T _data;
};

#else

template<typename T, class SpGroup>
class s_private_memory
{
  static constexpr int dimensions = SpGroup::dimensions;
public:
  template <class SG = SpGroup,
            std::enable_if_t<detail::is_sp_group_v<SG>, int> = 0>
  [[deprecated("Use sycl::memory_environment() instead")]]
  ACPP_KERNEL_TARGET
  explicit s_private_memory(const SpGroup& grp)
  : _data{new T [grp.get_logical_local_linear_range()]}, _grp{grp}
  {}

  s_private_memory(const s_private_memory&) = delete;
  s_private_memory& operator=(const s_private_memory&) = delete;

  [[deprecated("Use sycl::memory_environment() instead")]]
  ACPP_KERNEL_TARGET
  T& operator()(const detail::sp_item<dimensions>& idx) noexcept
  {
    return get(idx.get_local_id(_grp), idx.get_local_range(_grp));
  }

private:
  std::unique_ptr<T []> _data;
  const SpGroup& _grp;

  ACPP_KERNEL_TARGET
  T &get(const id<dimensions> &id,
         const range<dimensions> &local_range) noexcept {
    return _data.get()[detail::linear_id<dimensions>::get(id, local_range)];
  }
};
#endif

}
}

#endif
