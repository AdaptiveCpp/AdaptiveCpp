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


#ifndef HIPSYCL_BUFFER_ALLOCATOR_HPP
#define HIPSYCL_BUFFER_ALLOCATOR_HPP

#include <cstddef>
#include <memory>
#include <type_traits>
#include <limits>
#include <utility>

#include "backend/backend.hpp"
#include "exception.hpp"

namespace hipsycl {
namespace sycl {


template <typename T>
using buffer_allocator = std::allocator<T>;

// ToDo Image allocator

#ifdef __NVCC__
namespace hipsycl {

template<class T>
class svm_allocator
{
public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_pointer = const T*;
  using const_reference = const T&;
  using size_type = size_t;
  using difference_type = ptrdiff_t;

  template <class U> struct rebind {
    using other = svm_allocator<U>;
  };

  using propagate_on_container_move_assignment = std::true_type;

  svm_allocator() noexcept {}

  svm_allocator (const svm_allocator& alloc) noexcept {}

  template <class U>
  svm_allocator (const svm_allocator<U>& alloc) noexcept {}

  pointer address ( reference x ) const noexcept
  { return &x; }

  const_pointer address ( const_reference x ) const noexcept
  { return &x; }

  pointer allocate(size_type n, const void* hint=0)
  {
    void* ptr = nullptr;
    cudaError_t result = cudaMallocManaged(&ptr, n * sizeof(T));

    if(result != cudaSuccess)
      throw memory_allocation_error{"SVM allocator: bad allocation", static_cast<hipError_t>(result)};

    return reinterpret_cast<pointer>(ptr);
  }

  void deallocate (pointer p, size_type n)
  {
    cudaError_t result = cudaFree(p);
    if(result != cudaSuccess)
      throw runtime_error{"SVM allocator: Could not free memory", static_cast<hipError_t>(result)};
  }

  size_type max_size() const noexcept
  {
    return std::numeric_limits<size_type>::max();
  }

  template <class U, class... Args>
  void construct (U* p, Args&&... args)
  {
    *p = U(std::forward<Args>(args)...);
  }

  template <class U>
  void destroy (U* p)
  { p->~U(); }
};

}
#endif

}
}


#endif 
