/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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
  HIPSYCL_KERNEL_TARGET explicit private_memory_access(const SpGroup &grp,
                                                       T *data)
      : _data{data}, _grp{grp} {}

  private_memory_access(const private_memory_access&) = delete;
  private_memory_access& operator=(const private_memory_access&) = delete;

  HIPSYCL_KERNEL_TARGET
  T& operator()(const detail::sp_item<dimensions>& idx) noexcept
  {
    return get(idx.get_local_id(_grp), idx.get_local_range(_grp));
  }

private:
  const SpGroup& _grp;
  T* _data;

  HIPSYCL_KERNEL_TARGET
  T &get(const sycl::id<dimensions> &id,
         const sycl::range<dimensions> &local_range) noexcept {
    return _data[detail::linear_id<dimensions>::get(id, local_range)];
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
  HIPSYCL_KERNEL_TARGET explicit s_private_memory(const SpGroup & grp)
  {}

  s_private_memory(const s_private_memory&) = delete;
  s_private_memory& operator=(const s_private_memory&) = delete;

  [[deprecated("Use sycl::memory_environment() instead")]]
  HIPSYCL_KERNEL_TARGET
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
  HIPSYCL_KERNEL_TARGET
  explicit s_private_memory(const SpGroup& grp)
  : _data{new T [grp.get_logical_local_linear_range()]}, _grp{grp}
  {}

  s_private_memory(const s_private_memory&) = delete;
  s_private_memory& operator=(const s_private_memory&) = delete;

  [[deprecated("Use sycl::memory_environment() instead")]]
  HIPSYCL_KERNEL_TARGET
  T& operator()(const detail::sp_item<dimensions>& idx) noexcept
  {
    return get(idx.get_local_id(_grp), idx.get_local_range(_grp));
  }

private:
  std::unique_ptr<T []> _data;
  const SpGroup& _grp;

  HIPSYCL_KERNEL_TARGET
  T &get(const id<dimensions> &id,
         const range<dimensions> &local_range) noexcept {
    return _data.get()[detail::linear_id<dimensions>::get(id, local_range)];
  }
};
#endif

}
}

#endif
