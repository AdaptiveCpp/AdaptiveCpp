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
#ifndef HIPSYCL_THREAD_HIERARCHY_HPP
#define HIPSYCL_THREAD_HIERARCHY_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "../id.hpp"
#include "../range.hpp"
#include "data_layout.hpp"


namespace hipsycl {
namespace sycl {
namespace detail {

#define HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(                                 \
    name, cuda_variable, hip_variable, sscp_variable, host_variable)           \
  ACPP_KERNEL_TARGET                                                        \
  inline int name() {                                                          \
    __acpp_backend_switch(return 0, return sscp_variable(),                    \
                                 return cuda_variable, return hip_variable)    \
  }

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lid_x, 
  threadIdx.x,
  hipThreadIdx_x,
  __acpp_sscp_get_local_id_x,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lid_y, 
  threadIdx.y,
  hipThreadIdx_y,
  __acpp_sscp_get_local_id_y,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lid_z, 
  threadIdx.z,
  hipThreadIdx_z,
  __acpp_sscp_get_local_id_z,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_gid_x,
  blockIdx.x,
  hipBlockIdx_x,
  __acpp_sscp_get_group_id_x,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_gid_y,
  blockIdx.y,
  hipBlockIdx_y,
  __acpp_sscp_get_group_id_y,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_gid_z,
  blockIdx.z,
  hipBlockIdx_z,
  __acpp_sscp_get_group_id_z,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lsize_x,
  blockDim.x,
  hipBlockDim_x,
  __acpp_sscp_get_local_size_x,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lsize_y,
  blockDim.y,
  hipBlockDim_y,
  __acpp_sscp_get_local_size_y,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_lsize_z,
  blockDim.z,
  hipBlockDim_z,
  __acpp_sscp_get_local_size_z,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_ngroups_x,
  gridDim.x,
  hipGridDim_x,
  __acpp_sscp_get_num_groups_x,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_ngroups_y,
  gridDim.y,
  hipGridDim_y,
  __acpp_sscp_get_num_groups_y,
  0)

HIPSYCL_DEFINE_BUILTIN_VARIABLE_QUERY(__acpp_get_ngroups_z,
  gridDim.z,
  hipGridDim_z,
  __acpp_sscp_get_num_groups_z,
  0)

#define __acpp_lid_x ::hipsycl::sycl::detail::__acpp_get_lid_x()
#define __acpp_lid_y ::hipsycl::sycl::detail::__acpp_get_lid_y()
#define __acpp_lid_z ::hipsycl::sycl::detail::__acpp_get_lid_z()

#define __acpp_gid_x ::hipsycl::sycl::detail::__acpp_get_gid_x()
#define __acpp_gid_y ::hipsycl::sycl::detail::__acpp_get_gid_y()
#define __acpp_gid_z ::hipsycl::sycl::detail::__acpp_get_gid_z()

#define __acpp_lsize_x ::hipsycl::sycl::detail::__acpp_get_lsize_x()
#define __acpp_lsize_y ::hipsycl::sycl::detail::__acpp_get_lsize_y()
#define __acpp_lsize_z ::hipsycl::sycl::detail::__acpp_get_lsize_z()

#define __acpp_ngroups_x ::hipsycl::sycl::detail::__acpp_get_ngroups_x()
#define __acpp_ngroups_y ::hipsycl::sycl::detail::__acpp_get_ngroups_y()
#define __acpp_ngroups_z ::hipsycl::sycl::detail::__acpp_get_ngroups_z()

#if ACPP_LIBKERNEL_COMPILER_SUPPORTS_HIP ||                                 \
    ACPP_LIBKERNEL_COMPILER_SUPPORTS_CUDA

#ifdef ACPP_LIBKERNEL_CUDA_NVCXX
  // warpSize is not constexpr with nvc++. Hardcode to 32
  // for now
  #define __acpp_warp_size 32
#else
  #define __acpp_warp_size warpSize
#endif

#endif



// The get_global_id_* and get_global_size_* functions 
// should only be used in the implementation of more 
// high-level functions in this file since they do
// not take into the transformation needed to map
// the fastest SYCL index to the fastest hardware index:
// Per SYCL spec, the highest dimension (e.g. dim=2 for 3D)
// is the fastest moving spec. In HIP/CUDA, it is x.
// Consequently, any id or range that is actually used
// must be reversed before it can be used in a performant manner!
inline ACPP_KERNEL_TARGET size_t get_global_id_x()
{
  return __acpp_gid_x * __acpp_lsize_x + __acpp_lid_x;
}

inline ACPP_KERNEL_TARGET size_t get_global_id_y()
{
  return __acpp_gid_y * __acpp_lsize_y + __acpp_lid_y;
}

inline ACPP_KERNEL_TARGET size_t get_global_id_z()
{
  return __acpp_gid_z * __acpp_lsize_z + __acpp_lid_z;
}

inline ACPP_KERNEL_TARGET size_t get_global_size_x()
{
  return __acpp_ngroups_x * __acpp_lsize_x;
}

inline ACPP_KERNEL_TARGET size_t get_global_size_y()
{
  return __acpp_ngroups_y * __acpp_lsize_y;
}

inline ACPP_KERNEL_TARGET size_t get_global_size_z()
{
  return __acpp_ngroups_z * __acpp_lsize_z;
}



template<int dimensions>
ACPP_KERNEL_TARGET
sycl::id<dimensions> get_local_id();

template<>
ACPP_KERNEL_TARGET
inline sycl::id<1> get_local_id<1>()
{ return sycl::id<1>(__acpp_lid_x); }

template<>
ACPP_KERNEL_TARGET
inline sycl::id<2> get_local_id<2>()
{ return sycl::id<2>(__acpp_lid_y, __acpp_lid_x); }

template<>
ACPP_KERNEL_TARGET
inline sycl::id<3> get_local_id<3>()
{ return sycl::id<3>(__acpp_lid_z, __acpp_lid_y, __acpp_lid_x); }

template<int dimensions>
ACPP_KERNEL_TARGET
sycl::id<dimensions> get_global_id();

template<>
ACPP_KERNEL_TARGET
inline sycl::id<1> get_global_id<1>()
{ return sycl::id<1>{get_global_id_x()}; }

template<>
ACPP_KERNEL_TARGET
inline sycl::id<2> get_global_id<2>()
{
  return sycl::id<2>{get_global_id_y(), get_global_id_x()};
}

template<>
ACPP_KERNEL_TARGET
inline sycl::id<3> get_global_id<3>()
{
  return sycl::id<3>{get_global_id_z(),
                    get_global_id_y(),
                    get_global_id_x()};
}

// For the sake of consistency, we also reverse group ids
template<int dimensions>
ACPP_KERNEL_TARGET
sycl::id<dimensions> get_group_id();

template<>
ACPP_KERNEL_TARGET
inline sycl::id<1> get_group_id<1>()
{ return sycl::id<1>(__acpp_gid_x); }

template<>
ACPP_KERNEL_TARGET
inline sycl::id<2> get_group_id<2>()
{
  return sycl::id<2>(__acpp_gid_y,
                     __acpp_gid_x);
}

template<>
ACPP_KERNEL_TARGET
inline sycl::id<3> get_group_id<3>()
{
  return sycl::id<3>(__acpp_gid_z,
                     __acpp_gid_y,
                     __acpp_gid_x);
}

template<int dimensions>
ACPP_KERNEL_TARGET
sycl::range<dimensions> get_grid_size();

template<>
ACPP_KERNEL_TARGET
inline sycl::range<1> get_grid_size<1>()
{
  return sycl::range<1>(__acpp_ngroups_x);
}

template<>
ACPP_KERNEL_TARGET
inline sycl::range<2> get_grid_size<2>()
{
  return sycl::range<2>(__acpp_ngroups_y, __acpp_ngroups_x);
}

template<>
ACPP_KERNEL_TARGET
inline sycl::range<3> get_grid_size<3>()
{
  return sycl::range<3>(__acpp_ngroups_z, __acpp_ngroups_y, __acpp_ngroups_x);
}


template<int dimensions>
ACPP_KERNEL_TARGET
sycl::range<dimensions> get_local_size();

template<>
ACPP_KERNEL_TARGET
inline sycl::range<1> get_local_size<1>()
{
  return sycl::range<1>(__acpp_lsize_x);
}

template<>
ACPP_KERNEL_TARGET
inline sycl::range<2> get_local_size<2>()
{
  return sycl::range<2>(__acpp_lsize_y, __acpp_lsize_x);
}

template<>
ACPP_KERNEL_TARGET
inline sycl::range<3> get_local_size<3>()
{
  return sycl::range<3>(__acpp_lsize_z, __acpp_lsize_y, __acpp_lsize_x);
}

template<int dimensions>
ACPP_KERNEL_TARGET
sycl::range<dimensions> get_global_size()
{
  return get_local_size<dimensions>() * get_grid_size<dimensions>();
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_global_size(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_size<1>(int dimension)
{
  return get_global_size_x();
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_size<2>(int dimension)
{
  return dimension == 0 ? get_global_size_y() : get_global_size_x();
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_size<3>(int dimension)
{
  switch(dimension)
  {
  case 0:
    return get_global_size_z();
  case 1:
    return get_global_size_y();
  case 2:
    return get_global_size_x();
  }
  return 1;
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_grid_size(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_grid_size<1>(int dimension)
{ return __acpp_ngroups_x; }

template<>
ACPP_KERNEL_TARGET
inline size_t get_grid_size<2>(int dimension)
{
  return dimension == 0 ? __acpp_ngroups_y : __acpp_ngroups_x;
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_grid_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __acpp_ngroups_z;
  case 1:
    return __acpp_ngroups_y;
  case 2:
    return __acpp_ngroups_x;
  }
  return 1;
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_local_size(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_size<1>(int dimension)
{ return __acpp_lsize_x; }

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_size<2>(int dimension)
{
  return dimension == 0 ? __acpp_lsize_y : __acpp_lsize_x;
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __acpp_lsize_z;
  case 1:
    return __acpp_lsize_y;
  case 2:
    return __acpp_lsize_x;
  }
  return 1;
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_global_id(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_id<1>(int dimension)
{ return get_global_id_x(); }

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_id<2>(int dimension)
{ return dimension==0 ? get_global_id_y() : get_global_id_x();}

template<>
ACPP_KERNEL_TARGET
inline size_t get_global_id<3>(int dimension)
{
  switch(dimension)
  {
  case 0:
    return get_global_id_z();
  case 1:
    return get_global_id_y();
  case 2:
    return get_global_id_x();
  }
  return 0;
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_local_id(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_id<1>(int dimension)
{ return __acpp_lid_x; }

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_id<2>(int dimension)
{ return dimension == 0 ? __acpp_lid_y : __acpp_lid_x; }

template<>
ACPP_KERNEL_TARGET
inline size_t get_local_id<3>(int dimension)
{
  switch(dimension)
  {
  case 0:
    return __acpp_lid_z;
  case 1:
    return __acpp_lid_y;
  case 2:
    return __acpp_lid_x;
  }
  return 0;
}

template<int dimensions>
ACPP_KERNEL_TARGET
inline size_t get_group_id(int dimension);

template<>
ACPP_KERNEL_TARGET
inline size_t get_group_id<1>(int dimension)
{
  return __acpp_gid_x;
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_group_id<2>(int dimension)
{
  return dimension == 0 ? __acpp_gid_y : __acpp_gid_x;
}

template<>
ACPP_KERNEL_TARGET
inline size_t get_group_id<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __acpp_gid_z;
  case 1:
    return __acpp_gid_y;
  case 2:
    return __acpp_gid_x;
  }
  return 0;
}

}
}
}

#endif
