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
#ifndef HIPSYCL_TYPES_HPP
#define HIPSYCL_TYPES_HPP

#include <vector>
#include <string>
#include <functional>
#include <thread>
#include <memory>
#include <exception>
#include <mutex>

#include "libkernel/detail/int_types.hpp"

namespace hipsycl {
namespace sycl {

#ifndef CL_SYCL_NO_STD_VECTOR
template<class T, class Alloc = std::allocator<T>>
using vector_class [[deprecated("Use std::vector instead")]] = std::vector<T, Alloc>;
#endif

#ifndef CL_SYCL_NO_STD_STRING
using string_class [[deprecated("Use std::string instead")]] = std::string;
#endif

#ifndef CL_SYCL_NO_STD_FUNCTION
template<class Func>
using function_class [[deprecated("Use std::function instead")]] = std::function<Func>;
#endif

#ifndef CL_SYCL_NO_STD_MUTEX
using mutex_class [[deprecated("Use std::mutex instead")]] = std::mutex;
#endif

#ifndef CL_SYCL_NO_STD_UNIQUE_PTR
template<class T>
using unique_ptr_class [[deprecated("Use std::unique_ptr instead")]] = std::unique_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_SHARED_PTR
template<class T>
using shared_ptr_class [[deprecated("Use std::shared_ptr instead")]] = std::shared_ptr<T>;
#endif

#ifndef CL_SYCL_NO_STD_WEAK_PTR
template<class T>
using weak_ptr_class [[deprecated("Use std::weak_ptr instead")]] = std::weak_ptr<T>;
#endif

#ifndef CL_SYCL_NO_HASH
template<class T>
using hash_class [[deprecated("Use std::hash instead")]] = std::hash<T>;
#endif

using exception_ptr_class [[deprecated("Use std::exception_ptr instead")]] = std::exception_ptr;


using exception_ptr [[deprecated("Use std::exception_ptr instead")]] = exception_ptr_class;
using exception_list = std::vector<std::exception_ptr>;
using async_handler = std::function<void(sycl::exception_list)>;

namespace detail::half_impl {
    class half;
}
using half = detail::half_impl::half;

// \todo Better use uint32_t etc
namespace detail {
// Define types in analogy to OpenCL cl_* types
using u_char = __acpp_uint8;
using u_short = __acpp_uint16;
using u_int = __acpp_uint32;
using u_long = __acpp_uint64;

using s_char = __acpp_int8;
using s_short = __acpp_int16;
using s_int = __acpp_int32;
using s_long = __acpp_int64;


using hp_float = sycl::half;
using sp_float = float;
using dp_float = double;
} //detail

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
