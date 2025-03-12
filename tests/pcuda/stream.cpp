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
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(pcuda_stream);


BOOST_AUTO_TEST_CASE(CreateDestroy) {
  pcudaStream_t s1, s2, s3;
  BOOST_CHECK(pcudaStreamCreate(&s1) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamCreateWithFlags(&s2, pcudaStreamNonBlocking) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamCreateWithPriority(&s3, pcudaStreamNonBlocking, 0) == pcudaSuccess);
  
  int* data;
  BOOST_CHECK(pcudaMallocManaged(&data, sizeof(int)*3) == pcudaSuccess);

  pcudaParallelFor(1,1, 0, s1, [=](){
    data[0] = 1;
  });

  pcudaParallelFor(1,1, 0, s2, [=](){
    data[1] = 2;
  });

  pcudaParallelFor(1,1, 0, s3, [=](){
    data[2] = 3;
  });

  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);

  BOOST_CHECK(data[0] == 1);
  BOOST_CHECK(data[1] == 2);
  BOOST_CHECK(data[2] == 3);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);

  BOOST_CHECK(pcudaStreamDestroy(s1) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamDestroy(s2) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamDestroy(s3) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(Synchronize) {
  pcudaStream_t s1;
  BOOST_CHECK(pcudaStreamCreate(&s1) == pcudaSuccess);
  
  int* data;
  BOOST_CHECK(pcudaMallocManaged(&data, sizeof(int)*2) == pcudaSuccess);

  pcudaParallelFor(1,1, 0, s1, [=](){
    data[0] = 42;
  });

  pcudaParallelFor(1,1, 0, 0, [=](){
    data[1] = 43;
  });

  BOOST_CHECK(pcudaStreamSynchronize(s1) == pcudaSuccess);
  BOOST_CHECK(data[0] == 42);
  BOOST_CHECK(pcudaStreamSynchronize(0) == pcudaSuccess);
  BOOST_CHECK(data[1] == 43);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);

  BOOST_CHECK(pcudaStreamDestroy(s1) == pcudaSuccess);
}

BOOST_AUTO_TEST_SUITE_END()
