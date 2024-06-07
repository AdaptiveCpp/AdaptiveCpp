/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

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
  HIPSYCL_KERNEL_TARGET                                                        \
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

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP ||                                 \
    HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA

#ifdef HIPSYCL_LIBKERNEL_CUDA_NVCXX
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
inline HIPSYCL_KERNEL_TARGET size_t get_global_id_x()
{
  return __acpp_gid_x * __acpp_lsize_x + __acpp_lid_x;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_id_y()
{
  return __acpp_gid_y * __acpp_lsize_y + __acpp_lid_y;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_id_z()
{
  return __acpp_gid_z * __acpp_lsize_z + __acpp_lid_z;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_x()
{
  return __acpp_ngroups_x * __acpp_lsize_x;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_y()
{
  return __acpp_ngroups_y * __acpp_lsize_y;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_z()
{
  return __acpp_ngroups_z * __acpp_lsize_z;
}



template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::id<dimensions> get_local_id();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<1> get_local_id<1>()
{ return sycl::id<1>(__acpp_lid_x); }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<2> get_local_id<2>()
{ return sycl::id<2>(__acpp_lid_y, __acpp_lid_x); }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<3> get_local_id<3>()
{ return sycl::id<3>(__acpp_lid_z, __acpp_lid_y, __acpp_lid_x); }

template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::id<dimensions> get_global_id();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<1> get_global_id<1>()
{ return sycl::id<1>{get_global_id_x()}; }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<2> get_global_id<2>()
{
  return sycl::id<2>{get_global_id_y(), get_global_id_x()};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<3> get_global_id<3>()
{
  return sycl::id<3>{get_global_id_z(),
                    get_global_id_y(),
                    get_global_id_x()};
}

// For the sake of consistency, we also reverse group ids
template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::id<dimensions> get_group_id();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<1> get_group_id<1>()
{ return sycl::id<1>(__acpp_gid_x); }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<2> get_group_id<2>()
{
  return sycl::id<2>(__acpp_gid_y,
                     __acpp_gid_x);
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<3> get_group_id<3>()
{
  return sycl::id<3>(__acpp_gid_z,
                     __acpp_gid_y,
                     __acpp_gid_x);
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::range<dimensions> get_grid_size();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<1> get_grid_size<1>()
{
  return sycl::range<1>(__acpp_ngroups_x);
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<2> get_grid_size<2>()
{
  return sycl::range<2>(__acpp_ngroups_y, __acpp_ngroups_x);
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<3> get_grid_size<3>()
{
  return sycl::range<3>(__acpp_ngroups_z, __acpp_ngroups_y, __acpp_ngroups_x);
}


template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::range<dimensions> get_local_size();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<1> get_local_size<1>()
{
  return sycl::range<1>(__acpp_lsize_x);
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<2> get_local_size<2>()
{
  return sycl::range<2>(__acpp_lsize_y, __acpp_lsize_x);
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<3> get_local_size<3>()
{
  return sycl::range<3>(__acpp_lsize_z, __acpp_lsize_y, __acpp_lsize_x);
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::range<dimensions> get_global_size()
{
  return get_local_size<dimensions>() * get_grid_size<dimensions>();
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline size_t get_global_size(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_global_size<1>(int dimension)
{
  return get_global_size_x();
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_global_size<2>(int dimension)
{
  return dimension == 0 ? get_global_size_y() : get_global_size_x();
}

template<>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
inline size_t get_grid_size(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_grid_size<1>(int dimension)
{ return __acpp_ngroups_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_grid_size<2>(int dimension)
{
  return dimension == 0 ? __acpp_ngroups_y : __acpp_ngroups_x;
}

template<>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size<1>(int dimension)
{ return __acpp_lsize_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size<2>(int dimension)
{
  return dimension == 0 ? __acpp_lsize_y : __acpp_lsize_x;
}

template<>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
inline size_t get_global_id(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_global_id<1>(int dimension)
{ return get_global_id_x(); }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_global_id<2>(int dimension)
{ return dimension==0 ? get_global_id_y() : get_global_id_x();}

template<>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
inline size_t get_local_id(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_id<1>(int dimension)
{ return __acpp_lid_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_id<2>(int dimension)
{ return dimension == 0 ? __acpp_lid_y : __acpp_lid_x; }

template<>
HIPSYCL_KERNEL_TARGET
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
HIPSYCL_KERNEL_TARGET
inline size_t get_group_id(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_group_id<1>(int dimension)
{
  return __acpp_gid_x;
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_group_id<2>(int dimension)
{
  return dimension == 0 ? __acpp_gid_y : __acpp_gid_x;
}

template<>
HIPSYCL_KERNEL_TARGET
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
