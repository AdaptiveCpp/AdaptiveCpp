/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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

#ifndef HIPSYCL_LIBKERNEL_BUILTIN_DISPATCH_HPP
#define HIPSYCL_LIBKERNEL_BUILTIN_DISPATCH_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"

#define HIPSYCL_DISPATCH_BUILTIN(name, ...)                                    \
  __hipsycl_if_target_hiplike(hiplike_builtins::name(__VA_ARGS__););           \
  __hipsycl_if_target_spirv(spirv_builtins::name(__VA_ARGS__););               \
  __hipsycl_if_target_host(host_builtins::name(__VA_ARGS__););
#define HIPSYCL_RETURN_DISPATCH_BUILTIN(name, ...)                             \
  __hipsycl_if_target_hiplike(return hiplike_builtins::name(__VA_ARGS__););    \
  __hipsycl_if_target_spirv(return spirv_builtins::name(__VA_ARGS__););        \
  __hipsycl_if_target_host(return host_builtins::name(__VA_ARGS__););


#endif
