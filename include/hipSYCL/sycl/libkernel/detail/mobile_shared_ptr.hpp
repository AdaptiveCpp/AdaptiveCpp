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
#include "hipSYCL/sycl/libkernel/host/host_backend.hpp"
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
  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr() {
    __hipsycl_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{nullptr};
    );
  }

  // Argument is ignored on device
  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr(std::shared_ptr<T> ptr){
    __hipsycl_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{ptr};
    );
  }

  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr(const mobile_shared_ptr& other){
    __hipsycl_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{other.get_shared_ptr_ref()};
    );
  }

  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr(mobile_shared_ptr&& other){
    __hipsycl_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{other.get_shared_ptr_ref()};
    );
  }

  HIPSYCL_UNIVERSAL_TARGET
  ~mobile_shared_ptr() {
    __hipsycl_if_target_host(
      get_shared_ptr_ref().~shared_ptr();
    );
  }

  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr<T>& operator=(const mobile_shared_ptr& other) {
    __hipsycl_if_target_host(
      get_shared_ptr_ref() = other.get_shared_ptr_ref();
    );

    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  mobile_shared_ptr<T>& operator=(mobile_shared_ptr&& other) {
    __hipsycl_if_target_host(
      get_shared_ptr_ref() = other.get_shared_ptr_ref();
    );

    return *this;
  }

  HIPSYCL_UNIVERSAL_TARGET
  const T* get() const
  { 
    __hipsycl_if_target_device(
      // Use sizeof(_data) to make sure it doesn't get optimized away
      return reinterpret_cast<T*>(sizeof(_data));
    );
    __hipsycl_if_target_host(
      return get_shared_ptr_ref().get(); 
    );
  }

  HIPSYCL_UNIVERSAL_TARGET
  T* get()
  { 
    __hipsycl_if_target_device(
      // Use sizeof(_data) to make sure it doesn't get optimized away
      return reinterpret_cast<T*>(sizeof(_data));
    );
    __hipsycl_if_target_host(
      return get_shared_ptr_ref().get(); 
    );
  }

  // We cannot make this function available on device, since
  // it would pull shared_ptr_class<T> into device code.
  HIPSYCL_HOST_TARGET
  shared_ptr_class<T> get_shared_ptr() const {
    __hipsycl_if_target_host(
      return get_shared_ptr_ref();
    );
    __hipsycl_if_target_device(
      return nullptr;
    );
  }


private:
  HIPSYCL_HOST_TARGET
  std::shared_ptr<T>& get_shared_ptr_ref() {
    return *reinterpret_cast<std::shared_ptr<T>*>(_data);
  }

  HIPSYCL_HOST_TARGET
  const std::shared_ptr<T>& get_shared_ptr_ref() const {
    return *reinterpret_cast<const std::shared_ptr<T>*>(_data);
  }

  // Do NOT use union {shared_ptr<T>} because that would spill the types
  // of all T used inside mobile_shared_ptr into device IR. This can be problematic
  // for SSCP, where some backends (SPIR-V!) may choke even if function pointers
  // are just mentioned in the type list in the LLVM module.
  // Because of this, we need to completely erase the type of T.
  char _data alignas(alignof(std::shared_ptr<T>)) [sizeof(std::shared_ptr<T>)];
};

}
}
}

#endif
