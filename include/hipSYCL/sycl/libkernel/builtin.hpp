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

#ifndef HIPSYCL_BUILTIN_HPP
#define HIPSYCL_BUILTIN_HPP

#include "hipSYCL/sycl/libkernel/backend.hpp"
#include <type_traits>
#include <cstdlib>

// This is pretty much a hack - at this point the math/builtin headers
// need a major refactoring
#ifdef HIPSYCL_PLATFORM_SPIRV
template<class T>
T __spirv_ocl___hipsycl_min(T a, T b){ return a < b ? a : b;}
template<class T>
T __spirv_ocl___hipsycl_max(T a, T b){ return a > b ? a : b;}
#else
template<class T>
T __hipsycl_min(T a, T b){ return a < b ? a : b;}
template<class T>
T __hipsycl_max(T a, T b){ return a > b ? a : b;}
#endif

namespace hipsycl {
namespace sycl {


#define HIPSYCL_PP_CONCATENATE_IMPL(a,b) a ## b
#define HIPSYCL_PP_CONCATENATE(a,b) HIPSYCL_PP_CONCATENATE_IMPL(a,b)


#ifdef HIPSYCL_PLATFORM_SPIRV
#define HIPSYCL_STD_FUNCTION(function_name) \
  HIPSYCL_PP_CONCATENATE(__spirv_ocl_, function_name)
#else
#define HIPSYCL_STD_FUNCTION(function_name) function_name
#endif

#define HIPSYCL_ENABLE_IF_INTEGRAL(template_param) \
  std::enable_if_t<std::is_integral<template_param>::value>* = nullptr

#define HIPSYCL_DEFINE_GENINTEGERN_FUNCTION(name, func) \
  template<class int_type, int N, \
           HIPSYCL_ENABLE_IF_INTEGRAL(int_type)> \
  HIPSYCL_KERNEL_TARGET \
  inline vec<int_type, N> name(const vec<int_type, N>& a) {\
    vec<int_type,N> result = a; \
    detail::transform_vector(result, \
                      [](int_type x){return func(x);}); \
    return result; \
  }


#define HIPSYCL_DEFINE_GENINTEGERN_BINARY_FUNCTION(name, func) \
  template<class int_type, int N, \
           HIPSYCL_ENABLE_IF_INTEGRAL(int_type)> \
  HIPSYCL_KERNEL_TARGET \
  inline vec<int_type, N> name(const vec<int_type, N>& a, \
                                 const vec<int_type, N>& b) {\
    return detail::binary_vector_operation(a,b,\
                          [](int_type x, int_type y){return func(x,y);}); \
  }

#define HIPSYCL_DEFINE_BUILTIN(name, func) \
  template<class T, HIPSYCL_ENABLE_IF_INTEGRAL(T)> \
  HIPSYCL_KERNEL_TARGET inline T name(T x) {return ::HIPSYCL_STD_FUNCTION(func)(x);}

#define HIPSYCL_DEFINE_BINARY_BUILTIN(name, func) \
  template<class T, HIPSYCL_ENABLE_IF_INTEGRAL(T)> \
  HIPSYCL_KERNEL_TARGET inline T name(T x,T y) {return ::HIPSYCL_STD_FUNCTION(func)(x,y);}


#define HIPSYCL_DEFINE_GENINTEGER_STD_FUNCTION(name, func) \
  HIPSYCL_DEFINE_BUILTIN(name, func) \
  HIPSYCL_DEFINE_GENINTEGERN_FUNCTION(name, ::HIPSYCL_STD_FUNCTION(func))


 #define HIPSYCL_DEFINE_GENINTEGER_BINARY_STD_FUNCTION(name, func) \
  HIPSYCL_DEFINE_BINARY_BUILTIN(name, func) \
  HIPSYCL_DEFINE_GENINTEGERN_BINARY_FUNCTION(name, ::HIPSYCL_STD_FUNCTION(func))


HIPSYCL_DEFINE_GENINTEGER_BINARY_STD_FUNCTION(min, __hipsycl_min)
HIPSYCL_DEFINE_GENINTEGER_BINARY_STD_FUNCTION(max, __hipsycl_max)

}
}


#endif
