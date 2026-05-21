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
  BOOST_CHECK(queue.AdaptiveCpp_extract_inorder_executor() != nullptr);

  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 1);
  }

  pcudaStream_t stream = sycl::AdaptiveCpp_pcuda::make_stream(queue);
  // Submit to *default stream*
  err = pcudaParallelFor(problem_size / group_size, group_size, [=](){
    auto item = sycl::AdaptiveCpp_pcuda::this_nd_item<1>();
    data[item.get_global_linear_id()] += 1;
  });
  BOOST_CHECK(err == pcudaSuccess);

  // stream should reference default stream
  err = pcudaStreamSynchronize(stream);
  BOOST_CHECK(err == pcudaSuccess);
  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 2);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(device) {
  sycl::queue q;
  auto sycl_dev = q.get_device();

  int backend;
  int platform;
  int dev;
  sycl::AdaptiveCpp_pcuda::make_pcuda_device_indices(sycl_dev, backend,
                                                     platform, dev);
  BOOST_CHECK(sycl::AdaptiveCpp_pcuda::make_sycl_device(backend, platform,
                                                        dev) == sycl_dev);
}


BOOST_AUTO_TEST_CASE(event) {
  sycl::queue q;
  BOOST_CHECK(sycl::AdaptiveCpp_pcuda::set_pcuda_device(q.get_device()) ==
              pcudaSuccess);
  
  int* data;
  pcudaEvent_t pevt;
  BOOST_CHECK(pcudaEventCreate(&pevt) == pcudaSuccess);

  BOOST_CHECK(pcudaMallocManaged(&data, sizeof(int)) == pcudaSuccess);
  pcudaParallelFor(1, 1, [=](){
    *data = 42;
  });
  pcudaEventRecord(pevt);
  sycl::event sevt = sycl::AdaptiveCpp_pcuda::make_event(pevt);
  sevt.wait();
  BOOST_CHECK(*data == 42);


  BOOST_CHECK(pcudaEventDestroy(pevt) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_SUITE_END()
