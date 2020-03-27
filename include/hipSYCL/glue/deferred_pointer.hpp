/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2020 Aksel Alpay
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

#ifndef HIPSYCL_DEFERRED_POINTER_HPP
#define HIPSYCL_DEFERRED_POINTER_HPP

#include <memory>
#include <vector>

namespace hipsycl {
namespace glue {

/// A pointer class that can be captured by kernels (for use in accessors)
/// whose actual address can be assigned later, after it has been
/// captured and is no longer directly accessible.
/// This is dony by hijacking the copy constructor: Whenever it is invoked,
/// if it has not yet been assigned an address, the class checks
/// the address it was constructed with. If it now contains a non-null
/// address, the deferred_pointer assumes this address.
/// To trigger this mechanism, kernel objects are explicitly copied once
/// right before kernel submission.

template<class T>
class deferred_pointer
{
  deferred_pointer(void** initial_ptr)
  : _ptr{reinterpret_cast<T*>(initial_ptr)}, _initialized{false} {}

  deferred_pointer(const deferred_pointer &other) {
    _ptr = other.ptr;
    _initialized = other.initialized;

    maybe_init();
  }

  T *get() const
  { return _ptr; }

private:
  
  T* _ptr;
  bool _initialized;

  void maybe_init() {
#ifndef SYCL_DEVICE_ONLY
    if (!_initialized && _ptr) {
      T *target_ptr = *reinterpret_cast<T**>(_ptr);
      if (target_ptr){
        _ptr = target_ptr;
        _initialized = true;
      }
    }
#endif
  }
};

} // glue
} // hipsycl


#endif
