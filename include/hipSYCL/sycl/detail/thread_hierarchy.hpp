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

#include "../id.hpp"
#include "../range.hpp"
#include "../backend/backend.hpp"
#include "data_layout.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

// The get_global_id_* and get_global_size_* functions 
// should only be used in the implementation of more 
// high-level functions in this file since they do
// not take into the transformation needed to map
// the fastest SYCL index to the fastest hardware index:
// Per SYCL spec, the highest dimension (e.g. dim=2 for 3D)
// is the fastest moving spec. In HIP/CUDA, it is x.
// Consequently, any id or range that is actually used
// must be reversed before it can be used in a performant manner!
inline __device__ size_t get_global_id_x()
{
  return hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
}

inline __device__ size_t get_global_id_y()
{
  return hipBlockIdx_y * hipBlockDim_y + hipThreadIdx_y;
}

inline __device__ size_t get_global_id_z()
{
  return hipBlockIdx_z * hipBlockDim_z + hipThreadIdx_z;
}

inline __device__ size_t get_global_size_x()
{
  return hipGridDim_x * hipBlockDim_x;
}

inline __device__ size_t get_global_size_y()
{
  return hipGridDim_y * hipBlockDim_y;
}

inline __device__ size_t get_global_size_z()
{
  return hipGridDim_z * hipBlockDim_z;
}



template<int dimensions>
__device__
sycl::id<dimensions> get_local_id();

template<>
__device__
inline sycl::id<1> get_local_id<1>()
{ return sycl::id<1>{hipThreadIdx_x}; }

template<>
__device__
inline sycl::id<2> get_local_id<2>()
{ return sycl::id<2>{hipThreadIdx_y, hipThreadIdx_x}; }

template<>
__device__
inline sycl::id<3> get_local_id<3>()
{ return sycl::id<3>{hipThreadIdx_z, hipThreadIdx_y, hipThreadIdx_x}; }

template<int dimensions>
__device__
sycl::id<dimensions> get_global_id();

template<>
__device__
inline sycl::id<1> get_global_id<1>()
{ return sycl::id<1>{get_global_id_x()}; }

template<>
__device__
inline sycl::id<2> get_global_id<2>()
{
  return sycl::id<2>{get_global_id_y(), get_global_id_x()};
}

template<>
__device__
inline sycl::id<3> get_global_id<3>()
{
  return sycl::id<3>{get_global_id_z(),
                    get_global_id_y(),
                    get_global_id_x()};
}

// For the sake of consistency, we also reverse group ids
template<int dimensions>
__device__
sycl::id<dimensions> get_group_id();

template<>
__device__
inline sycl::id<1> get_group_id<1>()
{ return sycl::id<1>{hipBlockIdx_x}; }

template<>
__device__
inline sycl::id<2> get_group_id<2>()
{
  return sycl::id<2>{hipBlockIdx_y,
               hipBlockIdx_x};
}

template<>
__device__
inline sycl::id<3> get_group_id<3>()
{
  return sycl::id<3>{hipBlockIdx_z,
                    hipBlockIdx_y,
                    hipBlockIdx_x};
}

template<int dimensions>
__device__
sycl::range<dimensions> get_grid_size();

template<>
__device__
inline sycl::range<1> get_grid_size<1>()
{
  return sycl::range<1>{hipGridDim_x};
}

template<>
__device__
inline sycl::range<2> get_grid_size<2>()
{
  return sycl::range<2>{hipGridDim_y, hipGridDim_x};
}

template<>
__device__
inline sycl::range<3> get_grid_size<3>()
{
  return sycl::range<3>{hipGridDim_z, hipGridDim_y, hipGridDim_x};
}


template<int dimensions>
__device__
sycl::range<dimensions> get_local_size();

template<>
__device__
inline sycl::range<1> get_local_size<1>()
{
  return sycl::range<1>{hipBlockDim_x};
}

template<>
__device__
inline sycl::range<2> get_local_size<2>()
{
  return sycl::range<2>{hipBlockDim_y, hipBlockDim_x};
}

template<>
__device__
inline sycl::range<3> get_local_size<3>()
{
  return sycl::range<3>{hipBlockDim_z, hipBlockDim_y, hipBlockDim_x};
}

template<int dimensions>
__device__
sycl::range<dimensions> get_global_size()
{
  return get_local_size<dimensions>() * get_grid_size<dimensions>();
}

template<int dimensions>
__device__
inline size_t get_global_size(int dimension);

template<>
__device__
inline size_t get_global_size<1>(int dimension)
{
  return get_global_size_x();
}

template<>
__device__
inline size_t get_global_size<2>(int dimension)
{
  return dimension == 0 ? get_global_size_y() : get_global_size_x();
}

template<>
__device__
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
__device__
inline size_t get_grid_size(int dimension);

template<>
__device__
inline size_t get_grid_size<1>(int dimension)
{ return hipGridDim_x; }

template<>
__device__
inline size_t get_grid_size<2>(int dimension)
{
  return dimension == 0 ? hipGridDim_y : hipGridDim_x;
}

template<>
__device__
inline size_t get_grid_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return hipGridDim_z;
  case 1:
    return hipGridDim_y;
  case 2:
    return hipGridDim_x;
  }
  return 1;
}

template<int dimensions>
__device__
inline size_t get_local_size(int dimension);

template<>
__device__
inline size_t get_local_size<1>(int dimension)
{ return hipBlockDim_x; }

template<>
__device__
inline size_t get_local_size<2>(int dimension)
{
  return dimension == 0 ? hipBlockDim_y : hipBlockDim_x;
}

template<>
__device__
inline size_t get_local_size<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return hipBlockDim_z;
  case 1:
    return hipBlockDim_y;
  case 2:
    return hipBlockDim_x;
  }
  return 1;
}

template<int dimensions>
__device__
inline size_t get_global_id(int dimension);

template<>
__device__
inline size_t get_global_id<1>(int dimension)
{ return get_global_id_x(); }

template<>
__device__
inline size_t get_global_id<2>(int dimension)
{ return dimension==0 ? get_global_id_y() : get_global_id_x();}

template<>
__device__
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
__device__
inline size_t get_local_id(int dimension);

template<>
__device__
inline size_t get_local_id<1>(int dimension)
{ return hipThreadIdx_x; }

template<>
__device__
inline size_t get_local_id<2>(int dimension)
{ return dimension == 0 ? hipThreadIdx_y : hipThreadIdx_x; }

template<>
__device__
inline size_t get_local_id<3>(int dimension)
{
  switch(dimension)
  {
  case 0:
    return hipThreadIdx_z;
  case 1:
    return hipThreadIdx_y;
  case 2:
    return hipThreadIdx_x;
  }
  return 0;
}

template<int dimensions>
__device__
inline size_t get_group_id(int dimension);

template<>
__device__
inline size_t get_group_id<1>(int dimension)
{
  return hipBlockIdx_x;
}

template<>
__device__
inline size_t get_group_id<2>(int dimension)
{
  return dimension == 0 ? hipBlockIdx_y : hipBlockIdx_x;
}

template<>
__device__
inline size_t get_group_id<3>(int dimension)
{
  switch (dimension)
  {
  case 0:
    return hipBlockIdx_z;
  case 1:
    return hipBlockIdx_y;
  case 2:
    return hipBlockIdx_x;
  }
  return 0;
}

/// Flips dimensions such that the range is consistent with the mapping
/// of SYCL index dimensions to backend dimensions.
/// When launching a SYCL kernel, grid and blocksize should be transformed
/// using this function.
template<int dimensions>
inline dim3 make_kernel_launch_range(dim3 range);

template<>
inline dim3 make_kernel_launch_range<1>(dim3 range)
{
  return dim3(range.x, 1, 1);
}

template<>
inline dim3 make_kernel_launch_range<2>(dim3 range)
{
  return dim3(range.y, range.x, 1);
}

template<>
inline dim3 make_kernel_launch_range<3>(dim3 range)
{
  return dim3(range.z, range.y, range.x);
}

}
}
}

#endif
