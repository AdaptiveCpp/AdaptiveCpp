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

template<int V0 = 1, int V1 = 1, int V2 = 1>
struct static_range {
  static constexpr int v0 = V0;
  static constexpr int v1 = V1;
  static constexpr int v2 = V2;

  static constexpr int linear_range() {
    return v0 * v1 * v2;
  }

  static constexpr bool is_known() {
    return linear_range() != 0;
  }
};

using unknown_static_range = static_range<-1, -1, -1>;

struct unity_nested_range {
  using range = static_range<1,1,1>;
  using next = unity_nested_range;
};

template<class StaticRange, class NextRange = unity_nested_range>
struct nested_range {
  using range = StaticRange;
  using next = NextRange;
};


template<int Dim, int Level, class NestedRangeT>
struct sp_property_descriptor{
  static constexpr int dimensions = Dim;
  static constexpr int level = Level;

  static constexpr auto make_next_level_descriptor(){
    return sp_property_descriptor<Dim, Level+1, typename NestedRangeT::next>{};
  }

  static constexpr bool has_scalar_fixed_group_size(){
    return NestedRangeT::range::linear_range() == 1;
  }

  static auto get_fixed_group_size(){
    using range = typename NestedRangeT::range;
    if constexpr(Level > 0 && range::is_known()) {
      if constexpr(Dim == 1) {
        return sycl::range{range::v0};
      } else if constexpr(Dim == 2) {
        return sycl::range{range::v0, range::v1};
      } else {
        return sycl::range{range::v0, range::v1, range::v2};
      }
    } else {
      return sycl::range<dimensions>{};
    }
  }
};


/// Property descriptor wrapper that can be used to specialize groups
/// based on additional information
template<class Specialization, class SpPropertyDescriptor>
struct specialized_sp_property_descriptor {
  static constexpr int dimensions = SpPropertyDescriptor::dimensions;
  static constexpr int level = SpPropertyDescriptor::dimensions;

  static constexpr auto make_next_level_descriptor(){
    using next_t = decltype(SpPropertyDescriptor::make_next_level_descriptor());
    return specialized_sp_property_descriptor<Specialization, next_t>{};
  }

  static constexpr bool has_scalar_fixed_group_size(){
    return SpPropertyDescriptor::has_scalar_fixed_group_size();
  }

  static auto get_fixed_group_size(){
    return SpPropertyDescriptor::get_fixed_group_size();
  }
};

class host_specialization {};
class no_specialization {};

template <class SpPropertyDescriptor>
using host_sp_property_descriptor =
    specialized_sp_property_descriptor<host_specialization,
                                       SpPropertyDescriptor>;

template<class PropertyDescriptor>
struct sp_property_descriptor_traits {
  using next_level_descriptor_t =
      decltype(PropertyDescriptor::make_next_level_descriptor());
  using specialization = no_specialization;
};

template <class Specialization, class PropertyDescriptor>
struct sp_property_descriptor_traits<
    specialized_sp_property_descriptor<Specialization, PropertyDescriptor>> {
  using next_level_descriptor_t =
      decltype(specialized_sp_property_descriptor<
               Specialization,
               PropertyDescriptor>::make_next_level_descriptor());
  using specialization = Specialization;
};

template<class PropertyDescriptor>
constexpr bool is_host_property_descriptor() {
  using spec = typename sp_property_descriptor_traits<
      PropertyDescriptor>::specialization;
  return std::is_same_v<spec, host_specialization>;
}

template <class PropertyDescriptor>
using sp_next_level_descriptor_t = typename sp_property_descriptor_traits<
    PropertyDescriptor>::next_level_descriptor_t;

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
    __hipsycl_if_target_host(
      return idx.get_global_id() - get_group_id() * get_logical_local_range();
    );
    __hipsycl_if_target_device(
      return _grp.get_local_id();
    );
  }

  linear_id_type
  get_logical_local_linear_id(const detail::sp_item<dimensions> &idx) const noexcept {
    return detail::linear_id<dimensions>::get(get_logical_local_id(idx),
                                              get_logical_local_range());
  }

  size_t get_logical_local_id(const detail::sp_item<dimensions> &idx,
                              int dimension) const noexcept {
    __hipsycl_if_target_host(
      return idx.get_global_id(dimension) -
             get_group_id(dimension) * get_logical_local_range(dimension);
    );
    __hipsycl_if_target_device(
      return _grp.get_local_id(dimension);
    );
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
    __hipsycl_if_target_host(
      return id_type{};
    );
    __hipsycl_if_target_device(
      return _grp.get_local_id();
    );
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_id(int dimension) const noexcept {
    __hipsycl_if_target_host(
      return 0;
    );
    __hipsycl_if_target_device(
      return _grp.get_local_id(dimension);
    );
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_id() const noexcept {
    __hipsycl_if_target_host(
      return 0;
    );
    __hipsycl_if_target_device(
      return _grp.get_local_linear_id();
    );
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
    __hipsycl_if_target_host(
      if constexpr(dimensions == 1) {
        return sycl::range{1};
      } else if constexpr(dimensions == 2) {
        return sycl::range{1,1};
      } else {
        return sycl::range{1,1,1};
      }
    );
    __hipsycl_if_target_device(
      return _grp.get_local_range();
    );
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_range(int dimension) const noexcept {
    __hipsycl_if_target_host(
      return 1;
    );
    __hipsycl_if_target_device(
      _grp.get_local_range(dimension);
    );
  }

  HIPSYCL_KERNEL_TARGET
  size_t get_physical_local_linear_range() const noexcept {
    __hipsycl_if_target_host(
      return 1;
    );
    __hipsycl_if_target_device(
      return _grp.get_local_linear_range();
    );
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

// Device specialization for sp_sub_group
template<class PropertyDescriptor>
class sp_sub_group {
public:
  static constexpr std::size_t dimensions = PropertyDescriptor::dimensions;

  using id_type = sycl::id<dimensions>;
  using range_type = sycl::range<dimensions>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;
  
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sp_sub_group(const id_type& global_offset) noexcept
  : _global_offset{global_offset} {}

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

private:
  const sycl::id<dimensions> _global_offset;
};

// Host specialization for sub group
template<class PropertyDescriptor>
class sp_sub_group<host_sp_property_descriptor<PropertyDescriptor>> {
public:
  static constexpr std::size_t dimensions = PropertyDescriptor::dimensions;

  using id_type = sycl::id<dimensions>;
  using range_type = sycl::range<dimensions>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;
  
  static constexpr memory_scope fence_scope = memory_scope::sub_group;

  sp_sub_group(const id_type &group_id, const range_type &num_groups,
               const id_type &global_offset) noexcept
      : _group_id{group_id}, _num_groups{num_groups}, _global_offset{
                                                          global_offset} {}

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
private:
  struct storage {
    sycl::range<Dim> global_range;

    static storage& get() {
      static thread_local storage state;
      return state;
    }
  };

public:
  HIPSYCL_KERNEL_TARGET
  static sycl::range<Dim> get_global_range() noexcept {
    __hipsycl_if_target_host(
      return storage::get().global_range;
    );
    __hipsycl_if_target_device(
      return detail::get_global_size<Dim>();
    );
  }

  HIPSYCL_KERNEL_TARGET
  static void configure_global_range(const sycl::range<Dim> &range) noexcept {
    __hipsycl_if_target_host(
      storage::get().global_range = range;
    );
  }
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

  static_assert(next_property_descriptor::has_scalar_fixed_group_size(),
      "Non-scalar sub-subgroups are currently unsupported.");

  // TODO: On CPU we could introduce another tiling level
  if constexpr(!is_host_property_descriptor<PropertyDescriptor>()) {
    // Since we are decaying to scalar groups, the global offset
    // of the new "groups" is just the global id of the work item
    // which can always be obtained from the global offset
    // and local id.
    __hipsycl_if_target_device(
      sycl::id<dim> subgroup_global_offset =
          get_group_global_id_offset(g) + g.get_physical_local_id();
      
      sp_scalar_group<next_property_descriptor> subgroup{
          g.get_physical_local_id(), g.get_physical_local_range(),
          subgroup_global_offset};
      
      f(subgroup);
    );
  } else {
    __hipsycl_if_target_host(
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
    );
  }
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
  
  if constexpr(is_host_property_descriptor<PropertyDescriptor>()){
    static_assert(is_host_property_descriptor<next_property_descriptor>(),
      "Host property descriptor cannot spawn device property descriptor");
  
    const auto subgroup_size = next_property_descriptor::get_fixed_group_size();
    const auto num_groups = g.get_logical_local_range() / subgroup_size;
  
    glue::host::iterate_range_tiles(
        g.get_logical_local_range(), subgroup_size, [&](const sycl::id<dim> &idx) {

          sp_sub_group<next_property_descriptor> subgroup{
              idx, num_groups, get_group_global_id_offset(g) + idx * subgroup_size};
          
          f(subgroup);
        });
  } else {
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
    
  }
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
  __hipsycl_if_target_device(
    f(make_sp_item(g.get_physical_local_id(),
                  get_group_global_id_offset(g) + g.get_physical_local_id(),
                  g.get_logical_local_range(),
      sp_global_kernel_state<PropertyDescriptor::dimensions>::get_global_range()));
  );
  __hipsycl_if_target_host(
    auto global_range = sp_global_kernel_state<
            PropertyDescriptor::dimensions>::get_global_range();

    glue::host::iterate_range_simd(
        g.get_logical_local_range(), [&](auto local_idx) noexcept {
          f(make_sp_item(local_idx, get_group_global_id_offset(g) + local_idx,
                        g.get_logical_local_range(), global_range));
        });
  );
}

template<class PropertyDescriptor, typename NestedF>
HIPSYCL_KERNEL_TARGET
void distribute_items(const sp_group<PropertyDescriptor>& g, NestedF&& f) noexcept {
  auto global_range = g.get_logical_local_range() * g.get_group_range();

  __hipsycl_if_target_device(
    f(make_sp_item(g.get_physical_local_id(),
                  get_group_global_id_offset(g) + g.get_physical_local_id(),
                  g.get_logical_local_range(), global_range));
  );
  __hipsycl_if_target_host(
    const auto group_offset = get_group_global_id_offset(g);
    const auto local_range = g.get_logical_local_range();

    glue::host::iterate_range_simd(
        local_range, [&] (const auto local_idx) noexcept {
          f(make_sp_item(local_idx, group_offset + local_idx, local_range,
                        global_range));
        });
  );
}


}

template<class PropertyDescriptor>
using s_group = detail::sp_group<PropertyDescriptor>;

// Core group algorithms for scoped parallelism model

// TODO: SPIR-V
template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_group<PropertyDescriptor>::fence_scope) {
  __hipsycl_if_target_hiplike(
    if (fence_scope == memory_scope::device) {
      __threadfence_system();
    }
    __syncthreads();
  );
  __hipsycl_if_target_spirv(/* todo */);
  __hipsycl_if_target_host(/* todo */);
}

template <class PropertyDescriptor>
HIPSYCL_KERNEL_TARGET inline void
group_barrier(const detail::sp_sub_group<PropertyDescriptor> &g,
              memory_scope fence_scope =
                  detail::sp_sub_group<PropertyDescriptor>::fence_scope) {

  __hipsycl_if_target_hiplike(
    if (fence_scope == memory_scope::device) {
      __threadfence_system();
    } else if (fence_scope == memory_scope::work_group) {
      __threadfence_block();
    }
  );
  __hipsycl_if_target_cuda(
    __syncwarp();
  );
  __hipsycl_if_target_spirv(/* todo */);
  __hipsycl_if_target_host(/* todo */);
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

  __hipsycl_if_target_hiplike(
    if (fence_scope == memory_scope::device) {
      __threadfence_system();
    } else if (fence_scope == memory_scope::work_group) {
      __threadfence_block();
    }
  );
}


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
