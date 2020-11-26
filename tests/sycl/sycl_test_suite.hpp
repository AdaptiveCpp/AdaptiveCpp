/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_SYCL_TEST_SUITE_HPP
#define HIPSYCL_SYCL_TEST_SUITE_HPP

#include <tuple>

#define BOOST_MPL_CFG_GPU_ENABLED // Required for nvcc
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list_c.hpp>
#include <boost/mpl/list.hpp>


#define SYCL_SIMPLE_SWIZZLES
#include <CL/sycl.hpp>

#include "../common/reset.hpp"

using test_dimensions = boost::mpl::list_c<int, 1, 2, 3>;


template<int dimensions, template<int D> class T>
void assert_array_equality(const T<dimensions>& a, const T<dimensions>& b) {
  if(dimensions >= 1) BOOST_REQUIRE(a[0] == b[0]);
  if(dimensions >= 2) BOOST_REQUIRE(a[1] == b[1]);
  if(dimensions == 3) BOOST_REQUIRE(a[2] == b[2]);
}

template <template<int D> class T, int dimensions>
auto make_test_value(const T<1>& a, const T<2>& b, const T<3>& c) {
  return std::get<dimensions - 1>(std::make_tuple(a, b, c));
}

// Helper type to construct unique kernel names for all instantiations of
// a templated test case.
template<typename T, int dimensions, typename extra=T>
struct kernel_name {};

#endif
