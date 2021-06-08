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

#ifndef HIPSYCL_TYPES_HPP
#define HIPSYCL_TYPES_HPP

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <memory>
#include <exception>
#include <mutex>

namespace hipsycl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
template<class T, class Alloc = std::allocator<T>>
using vector_class = std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
using string_class = std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
template<class Func>
using function_class = std::function<Func>;
#endif

#ifndef CL_SYCL_NO_STD_MUTEX
using mutex_class = std::mutex;
#endif

#ifndef CL_SYCL_NO_STD_UNIQUE_PTR
template<class T>
using unique_ptr_class = std::unique_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_SHARED_PTR
template<class T>
using shared_ptr_class = std::shared_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_WEAK_PTR
template<class T>
using weak_ptr_class = std::weak_ptr<T>;
#endif

#ifndef CL_SYCL_NO_HASH
template<class T>
using hash_class = std::hash<T>;
#endif

using exception_ptr_class = std::exception_ptr;


using exception_ptr = exception_ptr_class;
using exception_list = vector_class<exception_ptr>;
using async_handler = function_class<void(sycl::exception_list)>;

// \todo Better use uint32_t etc
namespace detail {
// Define types in analogy to OpenCL cl_* types
using u_char = unsigned char;
using u_short = unsigned short;
using u_int = unsigned;
using u_long = unsigned long long;

using s_char = char;
using s_short = short;
using s_int = int;
using s_long = long long;

// ToDo: Proper half type
using hp_float = u_short;
using sp_float = float;
using dp_float = double;
} //detail

using half = detail::hp_float;
} // sycl
} // hipsycl

// Only pull typedefs into global namespace if the OpenCL headers
// defining them haven't yet been pulled in
#ifndef CL_TARGET_OPENCL_VERSION
#ifdef HIPSYCL_DEFINE_OPENCL_TYPES
using cl_uchar  = sycl::detail::u_char;
using cl_ushort = sycl::detail::u_short;
using cl_uint   = sycl::detail::u_int;
using cl_ulong  = sycl::detail::u_long;

using cl_char  = sycl::detail::s_char;
using cl_short = sycl::detail::s_short;
using cl_int   = sycl::detail::s_int;
using cl_long  = sycl::detail::s_long;

using cl_float  = sycl::detail::sp_float;
using cl_double = sycl::detail::dp_float;
using cl_half   = sycl::detail::hp_float;
#endif
#endif

#endif
