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
  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr() {
    __acpp_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{nullptr};
    );
  }

  // Argument is ignored on device
  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr(std::shared_ptr<T> ptr){
    __acpp_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{ptr};
    );
  }

  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr(const mobile_shared_ptr& other){
    __acpp_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{other.get_shared_ptr_ref()};
    );
  }

  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr(mobile_shared_ptr&& other){
    __acpp_if_target_host(
      new (&get_shared_ptr_ref()) std::shared_ptr<T>{other.get_shared_ptr_ref()};
    );
  }

  ACPP_UNIVERSAL_TARGET
  ~mobile_shared_ptr() {
    __acpp_if_target_host(
      get_shared_ptr_ref().~shared_ptr();
    );
  }

  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr<T>& operator=(const mobile_shared_ptr& other) {
    __acpp_if_target_host(
      get_shared_ptr_ref() = other.get_shared_ptr_ref();
    );

    return *this;
  }

  ACPP_UNIVERSAL_TARGET
  mobile_shared_ptr<T>& operator=(mobile_shared_ptr&& other) {
    __acpp_if_target_host(
      get_shared_ptr_ref() = other.get_shared_ptr_ref();
    );

    return *this;
  }

  ACPP_UNIVERSAL_TARGET
  const T* get() const
  { 
    __acpp_if_target_device(
      // Use sizeof(_data) to make sure it doesn't get optimized away
      return reinterpret_cast<T*>(sizeof(_data));
    );
    __acpp_if_target_host(
      return get_shared_ptr_ref().get(); 
    );
  }

  ACPP_UNIVERSAL_TARGET
  T* get()
  { 
    __acpp_if_target_device(
      // Use sizeof(_data) to make sure it doesn't get optimized away
      return reinterpret_cast<T*>(sizeof(_data));
    );
    __acpp_if_target_host(
      return get_shared_ptr_ref().get(); 
    );
  }

  // We cannot make this function available on device, since
  // it would pull shared_ptr_class<T> into device code.
  ACPP_HOST_TARGET
  shared_ptr_class<T> get_shared_ptr() const {
    __acpp_if_target_host(
      return get_shared_ptr_ref();
    );
    __acpp_if_target_device(
      return nullptr;
    );
  }


private:
  ACPP_HOST_TARGET
  std::shared_ptr<T>& get_shared_ptr_ref() {
    return *reinterpret_cast<std::shared_ptr<T>*>(_data);
  }

  ACPP_HOST_TARGET
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
