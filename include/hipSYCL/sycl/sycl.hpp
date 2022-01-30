/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay
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

#ifndef HIPSYCL_SYCL_HPP
#define HIPSYCL_SYCL_HPP


// Use this macro to detect hipSYCL from SYCL code
#ifndef __HIPSYCL__
 #define __HIPSYCL__
#endif

#define SYCL_IMPLEMENTATION_HIPSYCL

#ifdef CL_SYCL_LANGUAGE_VERSION
 #undef CL_SYCL_LANGUAGE_VERSION
#endif
#ifdef SYCL_LANGUAGE_VERSION
 #undef SYCL_LANGUAGE_VERSION
#endif

#define CL_SYCL_LANGUAGE_VERSION 202003
#define SYCL_LANGUAGE_VERSION 202003
#define SYCL_FEATURE_SET_FULL

#include "extensions.hpp"

#include "libkernel/backend.hpp"
#include "libkernel/range.hpp"
#include "libkernel/id.hpp"
#include "libkernel/accessor.hpp"
#include "libkernel/nd_item.hpp"
#include "libkernel/multi_ptr.hpp"
#include "libkernel/group.hpp"
#include "libkernel/h_item.hpp"
#include "libkernel/sp_item.hpp"
#include "libkernel/sp_group.hpp"
#include "libkernel/sp_private_memory.hpp"
#include "libkernel/memory_environment.hpp"
#include "libkernel/private_memory.hpp"
#include "libkernel/local_memory.hpp"
#include "libkernel/vec.hpp"
#include "libkernel/builtins.hpp"
#include "libkernel/atomic.hpp"
#include "libkernel/atomic_ref.hpp"
#include "libkernel/stream.hpp"
#include "libkernel/sub_group.hpp"
#include "libkernel/group_traits.hpp"
#include "libkernel/memory.hpp"
#if !HIPSYCL_LIBKERNEL_IS_DEVICE_PASS_SPIRV
 // Not yet supported for SPIR-V
 #include "libkernel/group_functions.hpp"
 #include "libkernel/group_functions_alias.hpp"
#endif
#include "libkernel/functional.hpp"
#include "libkernel/reduction.hpp"

#include "version.hpp"
#include "types.hpp"
#include "exception.hpp"
#include "device_selector.hpp"
#include "device.hpp"
#include "platform.hpp"
#include "queue.hpp"
#include "program.hpp"
#include "kernel.hpp"
#include "buffer.hpp"
#include "usm.hpp"
#include "backend.hpp"
#include "backend_interop.hpp"
#include "interop_handle.hpp"
#include "buffer_explicit_behavior.hpp"

#endif

