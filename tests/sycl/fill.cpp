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


#include "sycl_test_suite.hpp"
#include <boost/test/unit_test_suite.hpp>

template <int d>
void fill_test_helper(cl::sycl::id<d> offset = cl::sycl::id<d>{}) {
  namespace s = cl::sycl;

  cl::sycl::queue q;

  auto buff_size = make_test_value<s::range, d>({64}, {64, 64}, {64, 64, 64});
  s::buffer<s::id<d>, d> buff{buff_size};
  
  q.submit([&](s::handler& cgh){
    auto buff_acc = buff.template get_access<s::access::mode::discard_write>(cgh);
    cgh.parallel_for<kernel_name<class fill_init_kernel, d>>(buff_size,
      [=](s::id<d> idx){
      buff_acc[idx] = idx;
    });
  });

  auto fill_value = make_test_value<s::id, d>({3}, {3,3}, {3,3,3});
  q.submit([&](s::handler& cgh){
    if (offset != s::id<d>{}) {
      auto range = buff_size;
      for (int i = 0; i < d; ++i)
        range[i] -= offset[i];

      auto buff_acc = buff.template get_access<s::access::mode::write>(cgh, range, offset);
      cgh.fill(buff_acc, fill_value);
    } else {
      auto buff_acc = buff.template get_access<s::access::mode::write>(cgh);
      cgh.fill(buff_acc, fill_value);
    }

  });

  auto buff_host = buff.template get_access<s::access::mode::read>();

  size_t j_validation_range = (d >= 2) ? buff_size[1] : 1;
  size_t k_validation_range = (d == 3) ? buff_size[2] : 1;

  const auto after_offset = [&](s::id<d> idx) {
    if constexpr (d == 1)
      return offset[0] <= idx[0];
    if constexpr (d == 2) 
      return (offset[0] <= idx[0]) && (offset[1] <= idx[1]);
    if constexpr (d == 3)
      return (offset[0] <= idx[0]) && (offset[1] <= idx[1]) && (offset[2] <= idx[2]);
  };

  for(size_t i = 0; i < buff_size[0]; ++i)
    for(size_t j = 0; j < j_validation_range; ++j)
      for(size_t k = 0; k < k_validation_range; ++k)
      {        
        auto idx = make_test_value<s::id, d>({i}, {i,j}, {i,j,k});
        if (after_offset(idx))
          assert_array_equality(buff_host[idx], fill_value);
      }
}

BOOST_FIXTURE_TEST_SUITE(fill_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(fill_buffer, _dimensions,
  test_dimensions::type) {

  fill_test_helper<_dimensions::value>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(fill_with_offset, _dimensions,
                              test_dimensions::type) {
  namespace s = cl::sycl;
  auto offset = make_test_value<s::id, _dimensions::value>({8}, {8, 8}, {8, 8, 8});
  fill_test_helper<_dimensions::value>(offset);
}

BOOST_AUTO_TEST_SUITE_END()
