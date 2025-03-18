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


/// AdaptiveCpp understands normal libc math function calls in device code.
/// As such, we don't generally need to overload them with __device__ attribute
/// like clang CUDA or nvcc does.
/// However, there is a subtlety: CUDA defines additional overloads that are not
/// part of standard C++. In particular, it overloads functions for float (such that
/// e.g. ::sin() can invoke ::sinf() for a float argument).
/// In order to avoid a performance pitfall, we need to also expose these overloads.

#ifndef ACPP_PCUDA_MATH_HPP
#define ACPP_PCUDA_MATH_HPP

#include <cmath>

using std::sqrt;
using std::acos;
using std::asin;
using std::atan;
using std::atan2;
using std::ceil;
using std::cos;
using std::cosh;
using std::exp;
using std::fabs;
using std::floor;
using std::fmod;
using std::frexp;
using std::isinf;
using std::isfinite;
using std::isnan;
using std::ldexp;
using std::log;
using std::log10;
using std::modf;
using std::pow;
using std::signbit;
using std::sin;
using std::sinh;
using std::sqrt;
using std::tan;
using std::tanh;

#endif

