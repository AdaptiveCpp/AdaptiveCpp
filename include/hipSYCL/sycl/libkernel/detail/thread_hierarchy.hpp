/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef HIPSYCL_THREAD_HIERARCHY_HPP
#define HIPSYCL_THREAD_HIERARCHY_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#include "../id.hpp"
#include "../range.hpp"
#include "data_layout.hpp"

#if !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP &&                                \
    !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA &&                               \
    !HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV
#error "This file requires a device compiler"
#endif

namespace hipsycl {
namespace sycl {
namespace detail {

#ifndef SYCL_DEVICE_ONLY
// Define dummy values in case we are not in a device
// compilation pass. This makes it easier to use the
// functions from this file as we can call them
// without having to ifdef their usage.
#define __hipsycl_lid_x 0
#define __hipsycl_lid_y 0
#define __hipsycl_lid_z 0

#define __hipsycl_gid_x 0
#define __hipsycl_gid_y 0
#define __hipsycl_gid_z 0

#define __hipsycl_lsize_x 0
#define __hipsycl_lsize_y 0
#define __hipsycl_lsize_z 0

#define __hipsycl_ngroups_x 0
#define __hipsycl_ngroups_y 0
#define __hipsycl_ngroups_z 0
#else

#if HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_HIP

#define __hipsycl_lid_x hipThreadIdx_x
#define __hipsycl_lid_y hipThreadIdx_y
#define __hipsycl_lid_z hipThreadIdx_z

#define __hipsycl_gid_x hipBlockIdx_x
#define __hipsycl_gid_y hipBlockIdx_y
#define __hipsycl_gid_z hipBlockIdx_z

#define __hipsycl_lsize_x hipBlockDim_x
#define __hipsycl_lsize_y hipBlockDim_y
#define __hipsycl_lsize_z hipBlockDim_z

#define __hipsycl_ngroups_x hipGridDim_x
#define __hipsycl_ngroups_y hipGridDim_y
#define __hipsycl_ngroups_z hipGridDim_z

#elif HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_CUDA

#define __hipsycl_lid_x threadIdx.x
#define __hipsycl_lid_y threadIdx.y
#define __hipsycl_lid_z threadIdx.z

#define __hipsycl_gid_x blockIdx.x
#define __hipsycl_gid_y blockIdx.y
#define __hipsycl_gid_z blockIdx.z

#define __hipsycl_lsize_x blockDim.x
#define __hipsycl_lsize_y blockDim.y
#define __hipsycl_lsize_z blockDim.z

#define __hipsycl_ngroups_x gridDim.x
#define __hipsycl_ngroups_y gridDim.y
#define __hipsycl_ngroups_z gridDim.z

#elif HIPSYCL_LIBKERNEL_COMPILER_SUPPORTS_SPIRV

#define __hipsycl_lid_x __spirv_BuiltInLocalInvocationId.x
#define __hipsycl_lid_y __spirv_BuiltInLocalInvocationId.y
#define __hipsycl_lid_z __spirv_BuiltInLocalInvocationId.z

#define __hipsycl_gid_x __spirv_BuiltInWorkgroupId.x
#define __hipsycl_gid_y __spirv_BuiltInWorkgroupId.y
#define __hipsycl_gid_z __spirv_BuiltInWorkgroupId.z

#define __hipsycl_lsize_x __spirv_BuiltInWorkgroupSize.x
#define __hipsycl_lsize_y __spirv_BuiltInWorkgroupSize.y
#define __hipsycl_lsize_z __spirv_BuiltInWorkgroupSize.z

#define __hipsycl_ngroups_x __spirv_BuiltInNumWorkgroups.x
#define __hipsycl_ngroups_y __spirv_BuiltInNumWorkgroups.y
#define __hipsycl_ngroups_z __spirv_BuiltInNumWorkgroups.z

#endif
#endif // SYCL_DEVICE_ONLY


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
  return __hipsycl_gid_x * __hipsycl_lsize_x + __hipsycl_lid_x;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_id_y()
{
  return __hipsycl_gid_y * __hipsycl_lsize_y + __hipsycl_lid_y;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_id_z()
{
  return __hipsycl_gid_z * __hipsycl_lsize_z + __hipsycl_lid_z;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_x()
{
  return __hipsycl_ngroups_x * __hipsycl_lsize_x;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_y()
{
  return __hipsycl_ngroups_y * __hipsycl_lsize_y;
}

inline HIPSYCL_KERNEL_TARGET size_t get_global_size_z()
{
  return __hipsycl_ngroups_z * __hipsycl_lsize_z;
}



template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::id<dimensions> get_local_id();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<1> get_local_id<1>()
{ return sycl::id<1>{__hipsycl_lid_x}; }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<2> get_local_id<2>()
{ return sycl::id<2>{__hipsycl_lid_y, __hipsycl_lid_x}; }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<3> get_local_id<3>()
{ return sycl::id<3>{__hipsycl_lid_z, __hipsycl_lid_y, __hipsycl_lid_x}; }

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
{ return sycl::id<1>{__hipsycl_gid_x}; }

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<2> get_group_id<2>()
{
  return sycl::id<2>{__hipsycl_gid_y,
                     __hipsycl_gid_x};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::id<3> get_group_id<3>()
{
  return sycl::id<3>{__hipsycl_gid_z,
                     __hipsycl_gid_y,
                     __hipsycl_gid_x};
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::range<dimensions> get_grid_size();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<1> get_grid_size<1>()
{
  return sycl::range<1>{__hipsycl_ngroups_x};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<2> get_grid_size<2>()
{
  return sycl::range<2>{__hipsycl_ngroups_y, __hipsycl_ngroups_x};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<3> get_grid_size<3>()
{
  return sycl::range<3>{__hipsycl_ngroups_z, __hipsycl_ngroups_y, __hipsycl_ngroups_x};
}


template<int dimensions>
HIPSYCL_KERNEL_TARGET
sycl::range<dimensions> get_local_size();

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<1> get_local_size<1>()
{
  return sycl::range<1>{__hipsycl_lsize_x};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<2> get_local_size<2>()
{
  return sycl::range<2>{__hipsycl_lsize_y, __hipsycl_lsize_x};
}

template<>
HIPSYCL_KERNEL_TARGET
inline sycl::range<3> get_local_size<3>()
{
  return sycl::range<3>{__hipsycl_lsize_z, __hipsycl_lsize_y, __hipsycl_lsize_x};
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
{ return __hipsycl_ngroups_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_grid_size<2>(int dimension)
{
  return dimension == 0 ? __hipsycl_ngroups_y : __hipsycl_ngroups_x;
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_grid_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __hipsycl_ngroups_z;
  case 1:
    return __hipsycl_ngroups_y;
  case 2:
    return __hipsycl_ngroups_x;
  }
  return 1;
}

template<int dimensions>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size(int dimension);

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size<1>(int dimension)
{ return __hipsycl_lsize_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size<2>(int dimension)
{
  return dimension == 0 ? __hipsycl_lsize_y : __hipsycl_lsize_x;
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __hipsycl_lsize_z;
  case 1:
    return __hipsycl_lsize_y;
  case 2:
    return __hipsycl_lsize_x;
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
{ return __hipsycl_lid_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_id<2>(int dimension)
{ return dimension == 0 ? __hipsycl_lid_y : __hipsycl_lid_x; }

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_local_id<3>(int dimension)
{
  switch(dimension)
  {
  case 0:
    return __hipsycl_lid_z;
  case 1:
    return __hipsycl_lid_y;
  case 2:
    return __hipsycl_lid_x;
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
  return __hipsycl_gid_x;
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_group_id<2>(int dimension)
{
  return dimension == 0 ? __hipsycl_gid_y : __hipsycl_gid_x;
}

template<>
HIPSYCL_KERNEL_TARGET
inline size_t get_group_id<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return __hipsycl_gid_z;
  case 1:
    return __hipsycl_gid_y;
  case 2:
    return __hipsycl_gid_x;
  }
  return 0;
}

}
}
}

#endif
