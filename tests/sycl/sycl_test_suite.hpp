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
