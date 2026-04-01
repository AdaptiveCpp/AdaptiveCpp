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

BOOST_FIXTURE_TEST_SUITE(kernel_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(struct_arg) {
  sycl::queue q{sycl::property::queue::in_order{}};

  constexpr unsigned size = 256;
  constexpr unsigned arr_size = size / 2;
  std::vector<int> out_data(size);

  struct foo {
    int A[arr_size];
  } bar;

  for (unsigned i = 0; i < arr_size; i++) {
    bar.A[i] = i;
  }

  int *ptr = sycl::malloc_device<int>(size, q);
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(sycl::range<1>(size),
                     [=](sycl::id<1> id) { ptr[id] = bar.A[id / 2]; });
  });
  q.memcpy(out_data.data(), ptr, size * sizeof(int));
  q.wait();

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(i / 2 == out_data[i]);
    if (i / 2 != out_data[i])
      break;
  }
  sycl::free(ptr, q);
}

BOOST_AUTO_TEST_CASE(array_offset_subtract) {
  sycl::queue q{sycl::property::queue::in_order{}};

  size_t size = 128;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  int offset = 4;
  int *in_ptr = sycl::malloc_device<int>(size, q);
  int *out_ptr = sycl::malloc_device<int>(size, q);

  q.memcpy(in_ptr, in_data.data(), size * sizeof(int));
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<1>(size),
        [=](sycl::id<1> id) {
          out_ptr[id] = in_ptr[offset - 1];
        });
  });
  q.memcpy(out_data.data(), out_ptr, size * sizeof(int));
  q.wait();

  const int result = offset - 1;
  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(result == out_data[i]);
    if (result != out_data[i])
      break;
  }

  sycl::free(in_ptr, q);
  sycl::free(out_ptr, q);
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
