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

#ifndef HIPSYCL_SP_GROUP_HPP
#define HIPSYCL_SP_GROUP_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/detail/data_layout.hpp"
#include "hipSYCL/sycl/libkernel/group.hpp"
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
#include "hipSYCL/sycl/libkernel/sp_item.hpp"
#include "hipSYCL/sycl/libkernel/sub_group.hpp"
#include "hipSYCL/glue/generic/host/iterate_range.hpp"
#include <type_traits>

namespace hipsycl {
namespace sycl {

namespace detail {


template<int Dim, int Level, int... KnownGroupSizeDivisors>
struct sp_property_descriptor{
  static constexpr int dimensions = Dim;
  static constexpr int level = Level;

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
  static constexpr auto make_next_level_descriptor(){
    if constexpr(Level == 0) {
      using desired_next_level_type_1d = sp_property_descriptor<Dim, Level+1, 16>;
      using desired_next_level_type_2d = sp_property_descriptor<Dim, Level+1, 4,4>;
      using desired_next_level_type_3d = sp_property_descriptor<Dim, Level+1, 2,2,2>;

      if constexpr(Dim == 1) {
        return construct_next_level_or_scalar_fallback<16>();
      } else if constexpr(Dim == 2) {
        return construct_next_level_or_scalar_fallback<4,4>();
      } else {
        return construct_next_level_or_scalar_fallback<2,2,2>();
      }
    } else {
      return construct_scalar_next_level();
    }
  }

#elif HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA || HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP
  static constexpr auto make_next_level_descriptor(){
    if constexpr(Level == 0 && Dim == 1) {
      return construct_next_level_or_scalar_fallback<warpSize>();
    } else {
      return construct_scalar_next_level();
    }
  }
#else
  static constexpr auto make_next_level_descriptor(){
    return construct_scalar_next_level();
  }
#endif

  static constexpr bool has_known_group_size_divisors(){
    return sizeof...(KnownGroupSizeDivisors) > 0;
  }

  static constexpr bool has_scalar_fixed_group_size(){
    if constexpr(!has_known_group_size_divisors()) {
      return true;
    } else {
      return (KnownGroupSizeDivisors * ...) == 1;
    }
  }

  static auto get_fixed_group_size(){
    if constexpr(Level > 0 && has_known_group_size_divisors()) {
      return sycl::range{KnownGroupSizeDivisors...};
    } else {
      return sycl::range<dimensions>{};
    }
  }

  static_assert(
      has_known_group_size_divisors(),
      "No information about the group size was made available. This makes it "
      "impossible to reason about how to decompose work groups");

private:
  template<int... SubgroupSizes>
  static constexpr bool is_subgroup_guaranteed_supported() {
    return ((KnownGroupSizeDivisors % SubgroupSizes == 0 ) && ...);
  }

  template<int... SubgroupSizes>
  static constexpr auto construct_next_level_or_scalar_fallback() {
    if constexpr(is_subgroup_guaranteed_supported<SubgroupSizes...>()){
      return sp_property_descriptor<Dim, Level+1, SubgroupSizes...>{};
    } else {
      return construct_scalar_next_level();
    }
  }

  static constexpr auto construct_scalar_next_level() {
    using type_1d = sp_property_descriptor<Dim, Level+1, 1>;
    using type_2d = sp_property_descriptor<Dim, Level+1, 1,1>;
    using type_3d = sp_property_descriptor<Dim, Level+1, 1,1,1>;

    if constexpr(Dim==1) {
      return type_1d{};
    } else if constexpr (Dim==2) {
      return type_2d{};
    } else {
      return type_3d{};
    }
  }
};

template<class PropertyDescriptor>
struct sp_property_descriptor_traits {
  using next_level_descriptor_t =
      decltype(PropertyDescriptor::make_next_level_descriptor());
};

template <class PropertyDescriptor>
using sp_next_level_descriptor_t = typename sp_property_descriptor_traits<
    PropertyDescriptor>::next_level_descriptor_t;

template<int... Sizes>
struct sp_sub_group_host_descriptor {
  static constexpr int dimensions = sizeof...(Sizes);

  static sycl::range<dimensions> get_tile_size() {
    return sycl::range<dimensions>{Sizes...};
  }
};

template <class PropertyDescriptor>
struct sp_group
{
  static constexpr int dimensions = PropertyDescriptor::dimensions;
  using id_type = typename group<dimensions>::id_type;
  using range_type = typename group<dimensions>::range_type;
  using linear_id_type = typename group<dimensions>::linear_id_type;
  
  static constexpr memory_scope fence_scope = memory_scope::work_group;

  HIPSYCL_KERNEL_TARGET
  sycl::id<dimensions> get_group_id() const noexcept {
    return _grp.get_group_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_id(int dimension) const noexcept {
    return _grp.get_group_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_id() const noexcept {
    return _grp.get_group_linear_id();
  }

  /// \return The number of groups
  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_group_range() const noexcept {
    return _grp.get_group_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const noexcept {
    return _grp.get_group_range(dimension);
  }

  /// \return The overall number of groups
  HIPSYCL_KERNEL_TARGET
  size_t get_group_linear_range() const noexcept {
    return _grp.get_group_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const noexcept {
    return get_group_id(dimension);
  }

  friend bool operator==(const sp_group<PropertyDescriptor> &lhs,
                         const sp_group<PropertyDescriptor> &rhs) noexcept {
    return lhs._grp == rhs._grp;
  }

  friend bool operator!=(const sp_group<PropertyDescriptor>& lhs,
                         const sp_group<PropertyDescriptor>& rhs) noexcept {
    return !(lhs == rhs);
  }

  id_type
  get_logical_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return idx.get_global_id() - get_group_id() * get_logical_local_range();
#else
    return _grp.get_local_id();
#endif
  }

  linear_id_type
  get_logical_local_linear_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return detail::linear_id<dimensions>::get(get_logical_local_id(idx),
                                              get_logical_local_range());
  }

  size_t get_logical_local_id(const detail::sp_item<dimensions> &idx,
                              int dimension) const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return idx.get_global_id(dimension) -
           get_group_id(dimension) * get_logical_local_range(dimension);
#else
    return _grp.get_local_id(dimension);
#endif
  }

  id_type get_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_id(idx);
  }

  linear_id_type
  get_local_linear_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_linear_id(idx);
  }

  size_t get_local_id(const detail::sp_item<dimensions> &idx,
                      int dimension) const noexcept {
    return get_logical_local_id(idx, dimension);
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const noexcept {
    return get_physical_local_id();
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id(int dimension) const noexcept {
    return get_physical_local_id(dimension);
  }

  [[deprecated("Use get_physical_local_linear_id()")]]
  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const noexcept {
    return get_physical_local_linear_id();
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_physical_local_id() const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return id_type{};
#else
    return _grp.get_local_id();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return 0;
#else
    return _grp.get_local_id(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_id() const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return 0;
#else
    return _grp.get_local_linear_id();
#endif
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_local_range() const noexcept {
    return get_logical_local_range();
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const noexcept {
    return get_logical_local_range(dimension);
  }

  [[deprecated("Use get_logical_local_linear_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_range() const noexcept {
    return get_logical_local_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_logical_local_range() const noexcept {
    return _grp.get_local_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const noexcept {
    return _grp.get_local_range(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_linear_range() const noexcept {
    return _grp.get_local_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_physical_local_range() const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    if constexpr(dimensions == 1) {
      return sycl::range{1};
    } else if constexpr(dimensions == 2) {
      return sycl::range{1,1};
    } else {
      return sycl::range{1,1,1};
    }
#else
    return _grp.get_local_range();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return 1;
#else
    _grp.get_local_range(dimension);
#endif
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_range() const noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return 1;
#else
    return _grp.get_local_linear_range();
#endif
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET device_event
  async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                        size_t numElements) const noexcept {
    return _grp.async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET device_event
  async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                        size_t numElements) const noexcept {
    return _grp.async_work_group_copy(dest, src, numElements);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET device_event
  async_work_group_copy(local_ptr<dataT> dest, global_ptr<dataT> src,
                        size_t numElements, size_t srcStride) const noexcept {
    return _grp.async_work_group_copy(dest, src, numElements, srcStride);
  }

  template <typename dataT>
  HIPSYCL_KERNEL_TARGET device_event
  async_work_group_copy(global_ptr<dataT> dest, local_ptr<dataT> src,
                        size_t numElements, size_t destStride) const noexcept {
    return _grp.async_work_group_copy(dest, src, numElements, destStride);
  }

  template <typename... eventTN>
  HIPSYCL_KERNEL_TARGET
  void wait_for(eventTN...) const noexcept {}

  HIPSYCL_KERNEL_TARGET
  bool leader() const noexcept {
    return _grp.leader();
  }

  // Does not need to be private to be non-user constructible since
  // group<> is not user-constructible and the user cannot get a group<>
  // object in scoped parallelism.
  HIPSYCL_KERNEL_TARGET
  sp_group(const group<dimensions>& grp)
  : _grp{grp} {}
private:

  group<dimensions> _grp;
};

template<class PropertyDescriptor>
class sp_scalar_group
{
public:
  static constexpr int dimensions = PropertyDescriptor::dimensions;

  using id_type = sycl::id<dimensions>;
  using range_type = sycl::range<dimensions>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;

  static constexpr memory_scope fence_scope = memory_scope::work_item;

  sp_scalar_group(const id_type &group_idx, const range_type &num_groups,
                  const id_type &global_offset) noexcept
      : _group_id{group_idx}, _num_groups{num_groups}, _global_offset{
                                                           global_offset} {}

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const noexcept {
    return _group_id;
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id(int dimension) const noexcept {
    return _group_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const noexcept {
    return detail::linear_id<dimensions>::get(get_group_id(),
                                              get_group_range());
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const noexcept {
    return get_group_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const noexcept {
    return _num_groups.size();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const noexcept {
    return _num_groups;
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const noexcept {
    return true;
  }

  id_type get_global_group_offset() const noexcept {
    return _global_offset;
  }

  id_type
  get_logical_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return id_type{};
  }

  linear_id_type get_logical_local_linear_id(
      const detail::sp_item<dimensions> &idx) const noexcept {
    return 0;
  }

  size_t get_logical_local_id(const detail::sp_item<dimensions> &idx,
                              int dimension) const noexcept {
    return 0;
  }

  id_type get_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_id(idx);
  }

  linear_id_type
  get_local_linear_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_linear_id(idx);
  }

  size_t get_local_id(const detail::sp_item<dimensions> &idx,
                      int dimension) const noexcept {
    return get_logical_local_id(idx, dimension);
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const noexcept {
    return get_physical_local_id();
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id(int dimension) const noexcept {
    return get_physical_local_id(dimension);
  }

  [[deprecated("Use get_physical_local_linear_id()")]]
  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const noexcept {
    return get_physical_local_linear_id();
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_physical_local_id() const noexcept {
    return id_type{};
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const noexcept {
    return 0;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_id() const noexcept {
    return 0;
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_local_range() const noexcept {
    return get_logical_local_range();
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const noexcept {
    return get_logical_local_range(dimension);
  }

  [[deprecated("Use get_logical_local_linear_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_range() const noexcept {
    return get_logical_local_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_logical_local_range() const noexcept {
    range_type r;
    for(int i = 0; i < dimensions; ++i)
      r[i] = 1;
    return r;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const noexcept {
    return 1;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_linear_range() const noexcept {
    return 1;
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_physical_local_range() const noexcept {
    if constexpr(dimensions == 1) {
      return sycl::range{1};
    } else if constexpr(dimensions == 2) {
      return sycl::range{1,1};
    } else {
      return sycl::range{1,1,1};
    }
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const noexcept {
    return 1;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_range() const noexcept {
    return 1;
  }
private:
  const id_type _group_id;
  const range_type _num_groups;
  const id_type _global_offset;
};

/// Subgroup implementation
/// * On CPU, relies on static tile sizes from PropertyDescriptor.
/// * On device, relies on sycl::sub_group. sub_group is a 1D object,
/// in 2D and 3D will return e.g. (1, subgroup_size) or (1,1,subgroup_size).
///   In general, this cannot tesselate a 2D/3D work group! Additionally,
///   the local range mapped to one sub_group can not even be represented 
///   by a 2d/3d rectangle! It is therefore recommended to use this class
///   for 1D kernels only.
template<class PropertyDescriptor>
class sp_sub_group {
public:
  static constexpr std::size_t dimensions = PropertyDescriptor::dimensions;

  using id_type = sycl::id<dimensions>;
  using range_type = sycl::range<dimensions>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;
  
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  id_type get_global_group_offset() const noexcept {
    return _global_offset;
  }

  linear_id_type get_logical_local_linear_id(
      const detail::sp_item<dimensions> &idx) const noexcept {
    return detail::linear_id<dimensions>::get(get_logical_local_id(idx),
                                              get_logical_local_range());
  }

  id_type get_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_id(idx);
  }

  linear_id_type
  get_local_linear_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_logical_local_linear_id(idx);
  }

  size_t get_local_id(const detail::sp_item<dimensions> &idx,
                      int dimension) const noexcept {
    return get_logical_local_id(idx, dimension);
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id() const noexcept {
    return get_physical_local_id();
  }

  [[deprecated("Use get_physical_local_id()")]]
  HIPSYCL_KERNEL_TARGET
  id_type get_local_id(int dimension) const noexcept {
    return get_physical_local_id(dimension);
  }

  [[deprecated("Use get_physical_local_linear_id()")]]
  HIPSYCL_KERNEL_TARGET
  linear_id_type get_local_linear_id() const noexcept {
    return get_physical_local_linear_id();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_id() const noexcept {
    return detail::linear_id<dimensions>::get(get_physical_local_id(),
                                              get_physical_local_range());
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_local_range() const noexcept {
    return get_logical_local_range();
  }

  [[deprecated("Use get_logical_local_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_range(int dimension) const noexcept {
    return get_logical_local_range(dimension);
  }

  [[deprecated("Use get_logical_local_linear_range()")]]
  HIPSYCL_KERNEL_TARGET
  size_t get_local_linear_range() const noexcept {
    return get_logical_local_linear_range();
  }

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST

  sp_sub_group(const id_type &group_id, const range_type &num_groups,
               const id_type &global_offset) noexcept
      : _group_id{group_id}, _num_groups{num_groups}, _global_offset{
                                                          global_offset} {}

  id_type
  get_logical_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return idx.get_global_id() - _global_offset;
  }

  size_t get_logical_local_id(const detail::sp_item<dimensions> &idx,
                              int dimension) const noexcept {
    return idx.get_global_id(dimension) - _global_offset[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_physical_local_id() const noexcept {
    return id_type{};
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const noexcept {
    return 0;
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_logical_local_range() const noexcept {
    return PropertyDescriptor::get_fixed_group_size();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const noexcept {
    return get_logical_local_range()[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_linear_range() const noexcept {
    return get_logical_local_range().size();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_physical_local_range() const noexcept {
    return range_type{};
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const noexcept {
    return 1;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_range() const noexcept {
    return 1;
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const noexcept {
    return _group_id;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_id(int dimension) const noexcept {
    return _group_id[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  size_t operator[](int dimension) const noexcept {
    return get_group_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const noexcept {
    return detail::linear_id<dimensions>::get(get_group_id(),
                                              get_group_range());
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const noexcept {
    return _num_groups.size();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const noexcept {
    return _num_groups;
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const noexcept {
    return _num_groups[dimension];
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const noexcept {
    return get_local_linear_id() == 0;
  }
private:
  
  const id_type _group_id;
  const range_type _num_groups;
#else

  sp_sub_group(const id_type& global_offset) noexcept
  : _global_offset{global_offset} {}

  id_type
  get_logical_local_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return get_physical_local_id();
  }

  size_t get_logical_local_id(const detail::sp_item<dimensions> &idx,
                              int dimension) const noexcept {
    return get_physical_local_id(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  id_type get_physical_local_id() const noexcept {
    const size_t subgroup_lid = sycl::sub_group{}.get_local_linear_id();

    if constexpr(dimensions == 1) {
      return id_type{subgroup_lid};
    } else if constexpr(dimensions == 2){
      return id_type{0, subgroup_lid};
    } else {
      return id_type{0, 0, subgroup_lid};
    }
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const noexcept {
    if constexpr(dimensions == 1) {
      return sycl::sub_group{}.get_local_linear_id();
    } else {
      if(dimension == dimensions-1) {
        return sycl::sub_group{}.get_local_linear_id();
      } else {
        return 0;
      } 
    }
  }

  HIPSYCL_KERNEL_TARGET
  sycl::range<dimensions> get_logical_local_range() const noexcept {
    return get_physical_local_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_range(int dimension) const noexcept {
    return get_physical_local_range(dimension);
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_logical_local_linear_range() const noexcept {
    return get_physical_local_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_physical_local_range() const noexcept {
    if constexpr (dimensions == 1) {
      return range_type{sycl::sub_group{}.get_local_linear_range()};
    } else if constexpr (dimensions == 2) {
      return range_type{1, sycl::sub_group{}.get_local_linear_range()};
    } else if constexpr (dimensions == 3) {
      return range_type{1, 1, sycl::sub_group{}.get_local_linear_range()};
    } else {
      return range_type{};
    }
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const noexcept {
    if constexpr (dimensions == 1) {
      return sycl::sub_group{}.get_local_linear_range();
    } else {
      if(dimension == dimensions - 1){
        return sycl::sub_group{}.get_local_linear_range();
      } else {
        return 1;
      }
    }
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_range() const noexcept {
    return sycl::sub_group{}.get_local_linear_range();
  }


  HIPSYCL_KERNEL_TARGET
  id_type get_group_id() const noexcept {
    const size_t group_id = get_group_linear_id();

    if constexpr(dimensions == 1) {
      return id_type{group_id};
    } else if constexpr(dimensions == 2){
      return id_type{0, group_id};
    } else {
      return id_type{0, 0, group_id};
    }
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_id(int dimension) const noexcept {
    if constexpr(dimensions == 1){
      return get_group_linear_id();
    } else {
      if(dimension == dimensions - 1)
        return get_group_linear_id();
      else
        return 0;
    }
  }

  HIPSYCL_KERNEL_TARGET
  linear_id_type get_group_linear_id() const noexcept {
    return sycl::sub_group{}.get_group_linear_id();
  }

  HIPSYCL_KERNEL_TARGET
  linear_range_type get_group_linear_range() const noexcept {
    return sycl::sub_group{}.get_group_linear_range();
  }

  HIPSYCL_KERNEL_TARGET
  range_type get_group_range() const noexcept {
    return sycl::sub_group{}.get_group_range();
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_group_range(int dimension) const noexcept {
    if(dimension == dimensions - 1)
      return sycl::sub_group{}.get_group_linear_range();
    else
      return 1;
  }

  HIPSYCL_KERNEL_TARGET
  bool leader() const noexcept {
    return sycl::sub_group{}.leader();
  }

#endif
private:
  const sycl::id<dimensions> _global_offset;
};

template<class G>
struct is_sp_group : public std::false_type {};

template<class PropertyDescriptor>
struct is_sp_group<sp_group<PropertyDescriptor>> : public std::true_type {};

template<class PropertyDescriptor>
struct is_sp_group<sp_scalar_group<PropertyDescriptor>> : public std::true_type {};

template<class PropertyDescriptor>
struct is_sp_group<sp_sub_group<PropertyDescriptor>> : public std::true_type {};

template<class G>
inline constexpr bool is_sp_group_v = is_sp_group<G>::value;

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET
auto get_group_global_id_offset(
    const sp_group<PropertyDescriptor> &g) noexcept {
  return g.get_group_id() * g.get_logical_local_range();
}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET
auto get_group_global_id_offset(
    const sp_sub_group<PropertyDescriptor> &g) noexcept {
  return g.get_global_group_offset();
}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET
auto get_group_global_id_offset(
    const sp_scalar_group<PropertyDescriptor> &g) noexcept {
  return g.get_global_group_offset();
}


template<int Dim>
class sp_global_kernel_state {
public:
  HIPSYCL_KERNEL_TARGET
  static sycl::range<Dim> get_global_range() noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    return get()._global_range;
#else
    return detail::get_global_size<Dim>();
#endif
  }

  HIPSYCL_KERNEL_TARGET
  static void configure_global_range(const sycl::range<Dim> &range) noexcept {
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
    get()._global_range = range;
#endif
  }

private:
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST
  HIPSYCL_KERNEL_TARGET
  static sp_global_kernel_state& get() {
    static thread_local sp_global_kernel_state state;
    return state;
  }

  sycl::range<Dim> _global_range;
#endif
};

/// Subdivide a scalar group - this will always result in another
/// scalar group at the same position in the global iteration space.
/// However, the group id of the calling work item will then be 0, which
/// is not the case for a scalar group that was created from a non-scalar
/// group.
template<class PropertyDescriptor, class NestedF>
HIPSYCL_KERNEL_TARGET
inline void subdivide_group(
  const sp_scalar_group<PropertyDescriptor>& g, NestedF f) noexcept {
  
  // The next level when we already have a scalar group is a group
  // that is in a 1-element group iteration space and has id 0.
  //
  // The offset of the items in the global iteration space remains unchanged.
  constexpr int dim = sp_scalar_group<PropertyDescriptor>::dimensions;
  using next_property_descriptor =
      sp_next_level_descriptor_t<PropertyDescriptor>;

  sp_scalar_group<next_property_descriptor> subgroup{
      sycl::id<dim>{}, g.get_logical_local_range(), get_group_global_id_offset(g)};
  f(subgroup);
}

/// Subdivide a subgroup. Currently, this subdivides
/// into scalar groups on both device and host.
template<class PropertyDescriptor, class NestedF>
HIPSYCL_KERNEL_TARGET
inline  void subdivide_group(
  const sp_sub_group<PropertyDescriptor>& g, NestedF f) noexcept {
 
  constexpr int dim = sp_sub_group<PropertyDescriptor>::dimensions;
  using next_property_descriptor =
      sp_next_level_descriptor_t<PropertyDescriptor>;

  // TODO: On CPU we could introduce another tiling level
#ifdef SYCL_DEVICE_ONLY
  // Since we are decaying to scalar groups, the global offset
  // of the new "groups" is just the global id of the work item
  // which can always be obtained from the global offset
  // and local id.

  static_assert(next_property_descriptor::has_scalar_fixed_group_size(),
    "Non-scalar sub-subgroups are currently unimplemented.");

  sycl::id<dim> subgroup_global_offset =
      get_group_global_id_offset(g) + g.get_physical_local_id();
  
  sp_scalar_group<next_property_descriptor> subgroup{
      g.get_physical_local_id(), g.get_physical_local_range(),
      subgroup_global_offset};
  
  f(subgroup);
#else
  static_assert(next_property_descriptor::has_scalar_fixed_group_size(),
    "Non-scalar sub-subgroups are currently unsupported.");

  // On CPU, we need to iterate now across all elements of this subgroup
  // to construct scalar groups.
  if constexpr(next_property_descriptor::has_scalar_fixed_group_size()){
    glue::host::iterate_range_simd(
        g.get_logical_local_range(), [&](const sycl::id<dim> &idx) noexcept {
          sp_scalar_group<next_property_descriptor> subgroup{
              idx, g.get_logical_local_range(),
              get_group_global_id_offset(g) + idx};
          f(subgroup);
        });
  } else {
    glue::host::iterate_range_tiles(g.get_logical_local_range(), 
      next_property_descriptor::get_fixed_group_size(), [&](sycl::id<dim>& idx){
        // TODO: Multi-Level static tiling on CPU
        //sp_sub_group<next_property_descriptor> subgroup{};
      });
  }
#endif
}

/// Subdivide a work group into sub_group
template<class PropertyDescriptor, class NestedF>
HIPSYCL_KERNEL_TARGET
inline  void subdivide_group(
  const sp_group<PropertyDescriptor>& g, NestedF f) noexcept {
 
  constexpr int dim = sp_group<PropertyDescriptor>::dimensions;
  using next_property_descriptor =
      sp_next_level_descriptor_t<PropertyDescriptor>;
  
  // Need to store global range to allow querying global range in items
  sp_global_kernel_state<PropertyDescriptor::dimensions>::configure_global_range(
    g.get_group_range() * g.get_logical_local_range());
  
#ifdef SYCL_DEVICE_ONLY
  // This if statement makes sure that we only expose subgroups
  // when we can actually decompose the work group into subgroups.
  // Currently, this only exposes subgroups in the 1D case to make sure
  // all range and id queries are well defined
  if constexpr(!next_property_descriptor::has_scalar_fixed_group_size()) {
    const size_t global_offset = get_group_global_id_offset(g)[0] +
                                 g.get_physical_local_linear_id() -
                                 sycl::sub_group{}.get_local_linear_id();

    sp_sub_group<next_property_descriptor> subgroup{sycl::id<1>{global_offset}};
    f(subgroup);

  } else {
    sycl::id<dim> subgroup_global_offset =
      get_group_global_id_offset(g) + g.get_physical_local_id();
    
    sp_scalar_group<next_property_descriptor> subgroup{g.get_physical_local_id(),
      g.get_physical_local_range(), subgroup_global_offset};
    f(subgroup);
  }
#else

  const auto subgroup_size = next_property_descriptor::get_fixed_group_size();
  const auto num_groups = g.get_logical_local_range() / subgroup_size;
  glue::host::iterate_range_tiles(
      g.get_logical_local_range(), subgroup_size, [&](const sycl::id<dim> &idx) {

        sp_sub_group<next_property_descriptor> subgroup{
            idx, num_groups, get_group_global_id_offset(g) + idx * subgroup_size};
        
        f(subgroup);
      });
#endif
}

template <class PropertyDescriptor, typename NestedF>
HIPSYCL_KERNEL_TARGET
void distribute_items(const sp_scalar_group<PropertyDescriptor> &g,
                      NestedF f) noexcept {
  f(make_sp_item(sycl::id<PropertyDescriptor::dimensions>{},
                 get_group_global_id_offset(g), g.get_logical_local_range(),
    sp_global_kernel_state<PropertyDescriptor::dimensions>::get_global_range()));
}

template <class PropertyDescriptor, typename NestedF>
HIPSYCL_KERNEL_TARGET
void distribute_items(const sp_sub_group<PropertyDescriptor> &g,
                      NestedF f) noexcept {
#ifdef SYCL_DEVICE_ONLY
  f(make_sp_item(g.get_physical_local_id(),
                 get_group_global_id_offset(g) + g.get_physical_local_id(),
                 g.get_logical_local_range(),
    sp_global_kernel_state<PropertyDescriptor::dimensions>::get_global_range()));
#else
  auto global_range = sp_global_kernel_state<
          PropertyDescriptor::dimensions>::get_global_range();

  glue::host::iterate_range_simd(
      g.get_logical_local_range(), [&](auto local_idx) noexcept {
        f(make_sp_item(local_idx, get_group_global_id_offset(g) + local_idx,
                       g.get_logical_local_range(), global_range));
      });
#endif
}

template<class PropertyDescriptor, typename NestedF>
HIPSYCL_KERNEL_TARGET
void distribute_items(const sp_group<PropertyDescriptor>& g, NestedF&& f) noexcept {
  auto global_range = g.get_logical_local_range() * g.get_group_range();

#ifdef SYCL_DEVICE_ONLY
  f(make_sp_item(g.get_physical_local_id(),
                 get_group_global_id_offset(g) + g.get_physical_local_id(),
                 g.get_logical_local_range(), global_range));
#else
  const auto group_offset = get_group_global_id_offset(g);
  const auto local_range = g.get_logical_local_range();

  glue::host::iterate_range_simd(
      local_range, [&] (const auto local_idx) noexcept {
        f(make_sp_item(local_idx, group_offset + local_idx, local_range,
                       global_range));
      });
#endif
}


}

template<class PropertyDescriptor>
using s_group = detail::sp_group<PropertyDescriptor>;

// Core group algorithms for scoped parallelism model

// TODO: SPIR-V
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HIP ||                                    \
    HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_group<PropertyDescriptor>::fence_scope) {
  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  }
  __syncthreads();
}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_sub_group<PropertyDescriptor> &g,
              memory_scope fence_scope) {

  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_CUDA
  __syncwarp();
#endif
}

// Direct overload instead of default argument for memory fence
// can be used to optimize the if statement away at compile time
template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_scalar_group<PropertyDescriptor> &g) {}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_scalar_group<PropertyDescriptor> &g,
              memory_scope fence_scope) {

  if (fence_scope == memory_scope::device) {
    __threadfence_system();
  } else if (fence_scope == memory_scope::work_group) {
    __threadfence_block();
  }
}

#endif

#if HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_HOST

// TODO Handle system fence scope
template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_group<PropertyDescriptor>::fence_scope) {}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_sub_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_sub_group<PropertyDescriptor>::fence_scope) {}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_scalar_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_scalar_group<PropertyDescriptor>::fence_scope) {}

#endif

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void distribute_items(const Group &g,
                                                   Func f) noexcept {
  detail::distribute_items(g, f);
}

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void distribute_items_and_wait(const Group &g,
                                                            Func f) noexcept {
  detail::distribute_items(g, f);
  group_barrier(g);
}

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void distribute_groups(const Group &g,
                                                    Func f) noexcept {
  detail::subdivide_group(g, f);
}

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void distribute_groups_and_wait(const Group &g,
                                                             Func f) noexcept {
  detail::subdivide_group(g, f);
  group_barrier(g);
}

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void single_item(const Group &g, Func f) noexcept {
  if (g.leader())
    f();
}

template <class Group, class Func,
          std::enable_if_t<detail::is_sp_group_v<std::decay_t<Group>>, int> = 0>
HIPSYCL_KERNEL_TARGET inline void single_item_and_wait(const Group &g,
                                                       Func f) noexcept {
  if (g.leader())
    f();
  group_barrier(g);
}

} // namespace sycl
}
#endif
