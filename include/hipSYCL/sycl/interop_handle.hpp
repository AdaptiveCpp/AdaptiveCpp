/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay
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


#ifndef HIPSYCL_SYCL_INTEROP_HANDLE_HPP
#define HIPSYCL_SYCL_INTEROP_HANDLE_HPP

#include "access.hpp"
#include "backend.hpp"
#include "backend_interop.hpp"

#include "hipSYCL/runtime/executor.hpp"

namespace hipsycl {
namespace sycl {

class interop_handle {
public:
  interop_handle() = delete;
  interop_handle(rt::device_id assigned_device, void *kernel_launcher_params)
      : _dev{assigned_device}, _launcher_params{kernel_launcher_params},
        _executor{nullptr} {}

  interop_handle(rt::device_id assigned_device,
                 rt::backend_executor *executor)
      : _dev{assigned_device}, _launcher_params{nullptr}, _executor{executor} {}
  
  template <backend Backend, typename dataT, int dims, access::mode accessMode,
            access::target accessTarget, access::placeholder isPlaceholder>
  dataT* get_native_mem(const accessor<dataT, dims, accessMode, accessTarget,
                                isPlaceholder> &accessor) const {

    static_assert(accessTarget == access::target::global_buffer ||
                      accessTarget == access::target::host_buffer ||
                      accessTarget == access::target::constant_buffer,
                  "Invalid access target for accessor interop");

    return static_cast<dataT *>(
        glue::backend_interop<Backend>::get_native_mem(accessor));
  }

  // We don't have image types yet
  // 
  //template <backend Backend, typename dataT, int dims, access::mode accessMode,
  //          access::target accessTarget, access::placeholder isPlaceholder>
  //typename backend_traits<Backend>::template native_type<image>
  //get_native_mem(const accessor<dataT, dims, accessMode, accessTarget,
  //                              isPlaceholder> &imageAccessor) const;

  template <backend Backend>
  typename backend_traits<Backend>::template native_type<queue>
  get_native_queue() const noexcept {
    
    if(_launcher_params)
      return glue::backend_interop<Backend>::get_native_queue(_launcher_params);
    else if (_executor)
      return glue::backend_interop<Backend>::get_native_queue(_executor);

    HIPSYCL_DEBUG_WARNING
        << "interop_handle: Neither executor nor kernel launcher was provided, "
           "cannot construct native queue"
        << std::endl;
    
    return typename backend_traits<Backend>::template native_type<queue>{};
  }

  template <backend Backend>
  typename backend_traits<Backend>::template native_type<device>
  get_native_device() const noexcept {
    return glue::backend_interop<Backend>::get_native_device(device{_dev});
  }

  template <backend Backend>
  typename backend_traits<Backend>::template native_type<context>
  get_native_context() const noexcept {}

private:
  rt::device_id _dev;
  void *_launcher_params;
  rt::backend_executor *_executor;
};
} // namespace sycl
} // namespace hipsycl

#endif
