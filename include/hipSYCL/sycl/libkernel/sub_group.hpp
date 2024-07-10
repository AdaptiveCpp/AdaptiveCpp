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
#ifndef HIPSYCL_SUBGROUP_HPP
#define HIPSYCL_SUBGROUP_HPP

#include <cstdint>

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "detail/thread_hierarchy.hpp"
#include "id.hpp"
#include "range.hpp"
#include "memory.hpp"

#ifdef ACPP_LIBKERNEL_IS_DEVICE_PASS_SSCP
#include "sscp/builtins/subgroup.hpp"
#endif

namespace hipsycl {
namespace sycl {


class sub_group
{
public:
  using id_type = sycl::id<1>;
  using range_type = sycl::range<1>;
  using linear_id_type = uint32_t;
  using linear_range_type = uint32_t;

  static constexpr int dimensions = 1;
  static constexpr memory_scope fence_scope = memory_scope::sub_group;


  ACPP_KERNEL_TARGET
  id_type get_local_id() const {
    return id_type{get_local_linear_id()};
  }

  ACPP_KERNEL_TARGET
  linear_id_type get_local_linear_id() const {
    __acpp_backend_switch(
        return 0, 
        return __acpp_sscp_get_subgroup_local_id(),
        return local_tid() & get_warp_mask(),
        return local_tid() & get_warp_mask());
  }

  // always returns the maximum sub_group size
  ACPP_KERNEL_TARGET
  range_type get_local_range() const {
    return range_type{get_local_linear_range()};
  }

  // always returns the maximum sub_group size
  ACPP_KERNEL_TARGET
  linear_range_type get_local_linear_range() const {
    __acpp_backend_switch(
        return 1, 
        return __acpp_sscp_get_subgroup_size(),
        // TODO This is not actually correct for incomplete subgroups
        return __acpp_warp_size,
        return __acpp_warp_size);
  }

  ACPP_KERNEL_TARGET
  range_type get_max_local_range() const {
    __acpp_backend_switch(
        return range_type{1},
        return range_type{__acpp_sscp_get_subgroup_max_size()},
        return range_type{__acpp_warp_size},
        return range_type{__acpp_warp_size});
  }

  ACPP_KERNEL_TARGET
  id_type get_group_id() const {
    return id_type{get_group_linear_id()};
  }

  ACPP_KERNEL_TARGET
  linear_id_type get_group_linear_id() const {
    __acpp_backend_switch(
        return 0, // TODO This is probably incorrect
        return __acpp_sscp_get_subgroup_id(),
        return local_tid() >> (__ffs(__acpp_warp_size) - 1),
        return local_tid() >> (__ffs(__acpp_warp_size) - 1));
  }

  ACPP_KERNEL_TARGET
  linear_range_type get_group_linear_range() const {
    __acpp_backend_switch(
        return 1,
        return __acpp_sscp_get_num_subgroups(),
        return hiplike_num_subgroups(),
        return hiplike_num_subgroups());
  }

  ACPP_KERNEL_TARGET
  range_type get_group_range() const {
    return range_type{get_group_linear_range()};
  }

  [[deprecated]]
  ACPP_KERNEL_TARGET
  range_type get_max_group_range() const {
    return get_group_range();
  }

  ACPP_KERNEL_TARGET
  bool leader() const {
    return get_local_linear_id() == 0;
  }
private:
  int hiplike_num_subgroups() const {
    __acpp_if_target_hiplike(
        int local_range =
            __acpp_lsize_x * __acpp_lsize_y * __acpp_lsize_z;
        return (local_range + __acpp_warp_size - 1) / __acpp_warp_size;
    );
    return 0;
  }

  ACPP_KERNEL_TARGET
  int local_tid() const {
    __acpp_if_target_device(
      int tid = __acpp_lid_x 
              + __acpp_lid_y * __acpp_lsize_x 
              + __acpp_lid_z * __acpp_lsize_x * __acpp_lsize_y;
      return tid;
    );
    return 0;
  }

  ACPP_KERNEL_TARGET
  int get_warp_mask() const {
    // Assumes that __acpp_warp_size is a power of two
    __acpp_if_target_hiplike(
      return __acpp_warp_size - 1;
    );
    return 0;
  }
};

}
}

#endif
