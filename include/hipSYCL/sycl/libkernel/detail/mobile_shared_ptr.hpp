/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay
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

#ifndef HIPSYCL_MOBILE_SHARED_PTR_HPP
#define HIPSYCL_MOBILE_SHARED_PTR_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/types.hpp"

namespace hipsycl {
namespace sycl {
namespace detail {

/// A regular std::shared_ptr cannot be captured in kernels,
/// since it will in general depend on code that is not available
/// on device (C++ runtime, exceptions, CPU locks etc)
/// This implements a shared_ptr that can be captured
/// in device kernels. On CPU, it behaves like a regular shared_ptr.
/// While the usual shared_ptr functionality
/// regarding memory management is not supported on device,
/// and the managed memory content is not available on device,
/// it is guaranteed that the size of the object on host and device
/// are equal, such that it can be used as a kernel argument without
/// size mismatches.
template<class T>
class mobile_shared_ptr
{
public:
  mobile_shared_ptr() = default;

  // Not available on device
#ifndef SYCL_DEVICE_ONLY
  mobile_shared_ptr(shared_ptr_class<T> ptr)
  : _ptr{ptr}
  {}
#endif

  HIPSYCL_UNIVERSAL_TARGET
  const T* get() const
  { 
#ifdef SYCL_DEVICE_ONLY
    // Use sizeof(_buff) to make sure it doesn't get optimized away
    return reinterpret_cast<T*>(sizeof(_buff));
#else
    return _ptr.get(); 
#endif
  }

  HIPSYCL_UNIVERSAL_TARGET
  T* get()
  { 
#ifdef SYCL_DEVICE_ONLY
    // Use sizeof(_buff) to make sure it doesn't get optimized away
    return reinterpret_cast<T*>(sizeof(_buff));
#else
    return _ptr.get(); 
#endif
  }

  // We cannot make this function available on device, since
  // it would pull shared_ptr_class<T> into device code.
  HIPSYCL_HOST_TARGET
  shared_ptr_class<T> get_shared_ptr() const {
#ifndef SYCL_DEVICE_ONLY
    return _ptr;
#else
    return nullptr;
#endif
  }


private:
#ifdef SYCL_DEVICE_ONLY
  char _buff[sizeof(shared_ptr_class<T>)];
#else
  shared_ptr_class<T> _ptr;
#endif
};

}
}
}

#endif
