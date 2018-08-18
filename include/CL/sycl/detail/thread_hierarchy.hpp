/*
 * This file is part of SYCU, a SYCL implementation based CUDA/HIP
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

#ifndef SYCU_THREAD_HIERARCHY_HPP
#define SYCU_THREAD_HIERARCHY_HPP

#include "../id.hpp"
#include "../range.hpp"
#include "../backend/backend.hpp"
#include "../item.hpp"

namespace cl {
namespace sycl {
namespace detail {


template<int dimensions>
__device__
static id<dimensions> get_local_id();

template<>
__device__
id<1> get_local_id<1>()
{ return id<1>{hipThreadIdx_x}; }

template<>
__device__
id<2> get_local_id<2>()
{ return id<2>{hipThreadIdx_x, hipThreadIdx_y}; }

template<>
__device__
id<3> get_local_id<3>()
{ return id<3>{hipThreadIdx_x, hipThreadIdx_y, hipThreadIdx_z}; }

template<int dimensions>
__device__
static id<dimensions> get_global_id();

template<>
__device__
id<1> get_global_id<1>()
{ return id<1>{get_global_id_x()}; }

template<>
__device__
id<2> get_global_id<2>()
{
  return id<2>{get_global_id_x(),
               get_global_id_y()};
}

template<>
__device__
id<3> get_global_id<3>()
{
  return id<3>{get_global_id_x(),
               get_global_id_y(),
               get_global_id_z()};
}

template<int dimensions>
__device__
static id<dimensions> get_group_id();

template<>
__device__
id<1> get_group_id<1>()
{ return id<1>{hipBlockIdx_x}; }

template<>
__device__
id<2> get_group_id<2>()
{
  return id<2>{hipBlockIdx_x,
               hipBlockIdx_y};
}

template<>
__device__
id<3> get_group_id<3>()
{
  return id<3>{hipBlockIdx_x,
               hipBlockIdx_y,
               hipBlockIdx_z};
}

template<int dimensions>
__device__
static sycl::range<dimensions> get_grid_size();

template<>
__device__
sycl::range<1> get_grid_size<1>()
{
  return sycl::range<1>{hipGridDim_x};
}

template<>
__device__
sycl::range<2> get_grid_size<2>()
{
  return sycl::range<2>{hipGridDim_x, hipGridDim_y};
}

template<>
__device__
sycl::range<3> get_grid_size<3>()
{
  return sycl::range<3>{hipGridDim_x, hipGridDim_y, hipGridDim_z};
}


template<int dimensions>
__device__
static sycl::range<dimensions> get_local_size();

template<>
__device__
sycl::range<1> get_local_size<1>()
{
  return sycl::range<1>{hipBlockDim_x};
}

template<>
__device__
sycl::range<2> get_local_size<2>()
{
  return sycl::range<2>{hipBlockDim_x, hipBlockDim_y};
}

template<>
__device__
sycl::range<3> get_local_size<3>()
{
  return sycl::range<3>{hipBlockDim_x, hipBlockDim_y, hipBlockDim_z};
}

template<int dimensions>
__device__
static sycl::range<dimensions> get_global_size()
{
  return get_local_size<dimensions>() * get_grid_size<dimensions>();
}

__device__
static size_t get_global_size(int dimension)
{
  switch(dimension)
  {
  case 0:
    return hipBlockDim_x * hipGridDim_x;
  case 1:
    return hipBlockDim_y * hipGridDim_y;
  case 2:
    return hipBlockDim_z * hipGridDim_z;
  }
  return 1;
}

__device__
static size_t get_grid_size(int dimension)
{
  switch (dimension)
  {
  case 0:
    return hipGridDim_x;
  case 1:
    return hipGridDim_y;
  case 2:
    return hipGridDim_z;
  }
  return 1;
}

__device__
static size_t get_local_size(int dimension)
{
  switch(dimension)
  {
  case 0:
    return hipThreadIdx_x;
  case 1:
    return hipThreadIdx_y;
  case 2:
    return hipThreadIdx_z;
  }
  return 1;
}

__device__
static size_t get_global_id(int dimension)
{
  switch(dimension)
  {
  case 0:
    return get_global_id_x();
  case 1:
    return get_global_id_y();
  case 2:
    return get_global_id_z();
  }
  return 0;
}

__device__
static size_t get_local_id(int dimension)
{
  switch(dimension)
  {
  case 0:
    return hipThreadIdx_x;
  case 1:
    return hipThreadIdx_y;
  case 2:
    return hipThreadIdx_z;
  }
  return 0;
}

__device__
static size_t get_group_id(int dimension)
{
  switch (dimension)
  {
  case 0:
    return hipBlockIdx_x;
  case 1:
    return hipBlockIdx_y;
  case 2:
    return hipBlockIdx_z;
  }
  return 0;
}



}
}
}

#endif
