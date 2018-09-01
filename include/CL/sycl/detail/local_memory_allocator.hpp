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

#ifndef HIPSYCL_LOCAL_MEM_ALLOCATOR_HPP
#define HIPSYCL_LOCAL_MEM_ALLOCATOR_HPP

#include "../backend/backend.hpp"

#include <cstdlib>

namespace cl {
namespace sycl {

class device;

namespace detail {


class local_memory_allocator
{
public:
  using address = size_t;
  using smallest_type = int;

  // ToDo: Query max shared memory of device and check when allocating
  local_memory_allocator(const device&)
    : _num_allocated_bytes{0}
  {}

  template<class T>
  address alloc(size_t num_elements)
  {
    size_t num_bytes = num_elements * sizeof(T);

    size_t alignment = get_alignment<T>();

    size_t start_byte =
        alignment * ((get_allocation_size() + alignment - 1) / alignment);

    address start_address = start_byte;
    address end_address = start_address + num_bytes;

    _num_allocated_bytes = end_address;

    return start_address;
  }

  size_t get_allocation_size() const
  {
    return _num_allocated_bytes;
  }

  template<class T>
  size_t get_alignment() const
  {
    size_t alignment = std::max(sizeof(smallest_type), sizeof(T));
    // If T is not a multiple of an int in size (i.e, no power of 2
    // greater than 4), it is likely some weird data structure and not a vector
    // - in this case it's probably sufficient to align it by 32 bits.
    // ToDo: Check precise alignment rules for shared memory in CUDA and HIP
    if(sizeof(T) % sizeof(smallest_type) != 0)
      alignment = sizeof(smallest_type);

    return alignment;
  }
private:
  size_t _num_allocated_bytes;
};

extern __shared__ local_memory_allocator::smallest_type local_mem_data [];

class local_memory
{
public:
  using address = local_memory_allocator::address;

  template<class T>
  __device__
  static T* get_ptr(const address addr)
  {
    return reinterpret_cast<T*>(reinterpret_cast<char*>(local_mem_data) + addr);
  }
};

}
}
}

#endif
