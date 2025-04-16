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

#include <vector>

#include <pcuda.hpp>
#include <sycl/sycl.hpp>

#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(pcuda_interop);


BOOST_AUTO_TEST_CASE(nd_item) {
  int* data;
  int problem_size = 1024;
  int group_size = 128;
  BOOST_TEST(pcudaMallocManaged(&data, problem_size * sizeof(int)) == pcudaSuccess);

  for(int i = 0; i < problem_size; ++i)
    data[i] = i;

  auto err = pcudaParallelFor(problem_size / group_size, group_size, [=](){
    auto item = sycl::AdaptiveCpp_pcuda::this_nd_item<1>();
    data[item.get_global_linear_id()] += 1;
  });
  BOOST_CHECK(err == pcudaSuccess);

  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 1);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(queue) {
  int* data;
  int problem_size = 1024;
  int group_size = 128;
  BOOST_TEST(pcudaMallocManaged(&data, problem_size * sizeof(int)) == pcudaSuccess);

  for(int i = 0; i < problem_size; ++i)
    data[i] = i;

  auto err = pcudaParallelFor(problem_size / group_size, group_size, [=](){
    auto item = sycl::AdaptiveCpp_pcuda::this_nd_item<1>();
    data[item.get_global_linear_id()] += 1;
  });
  BOOST_CHECK(err == pcudaSuccess);

  auto queue = sycl::AdaptiveCpp_pcuda::make_queue(0);
  queue.wait();
  
  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 1);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_SUITE_END()
