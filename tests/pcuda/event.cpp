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


BOOST_AUTO_TEST_SUITE(pcuda_event);


BOOST_AUTO_TEST_CASE(CreateDestroy) {
  pcudaEvent_t e1, e2;
  BOOST_CHECK(pcudaEventCreate(&e1) == pcudaSuccess);
  BOOST_CHECK(pcudaEventCreateWithFlags(&e2, pcudaEventDisableTiming) == pcudaSuccess);
  

  BOOST_CHECK(pcudaEventDestroy(e1) == pcudaSuccess);
  BOOST_CHECK(pcudaEventDestroy(e2) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(Synchronize) {
  pcudaEvent_t e;

  BOOST_CHECK(pcudaEventCreate(&e) == pcudaSuccess);


  int* data;
  BOOST_CHECK(pcudaMallocManaged(&data, sizeof(int)) == pcudaSuccess);

  pcudaParallelFor(1,1,[=](){
    *data = 42;
  });

  BOOST_CHECK(pcudaEventRecord(e, 0) == pcudaSuccess);
  BOOST_CHECK(pcudaEventSynchronize(e) == pcudaSuccess);
  BOOST_CHECK(*data == 42);
  BOOST_CHECK(pcudaEventQuery(e) == pcudaSuccess);
  
  pcudaParallelFor(1,1,[=](){
    *data = 44;
  });

  BOOST_CHECK(pcudaEventRecord(e, 0) == pcudaSuccess);
  BOOST_CHECK(pcudaEventSynchronize(e) == pcudaSuccess);
  BOOST_CHECK(*data == 44);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
  BOOST_CHECK(pcudaEventDestroy(e) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(StreamWaitEvent) {
  pcudaStream_t s1, s2;
  BOOST_CHECK(pcudaStreamCreate(&s1) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamCreate(&s2) == pcudaSuccess);

  
  int* data;
  BOOST_CHECK(pcudaMallocManaged(&data, sizeof(int)) == pcudaSuccess);


  pcudaEvent_t e;
  BOOST_CHECK(pcudaEventCreate(&e) == pcudaSuccess);

  pcudaParallelFor(1,1, 0, s1, [=](){
    *data = 42;
  });

  BOOST_CHECK(pcudaEventRecord(e, s1) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamWaitEvent(s2, e) == pcudaSuccess);
  pcudaParallelFor(1,1, 0, s2, [=](){
    *data *= 2;
  });

  BOOST_CHECK(pcudaStreamSynchronize(s2) == pcudaSuccess);

  BOOST_CHECK(*data == 84);

  BOOST_CHECK(pcudaEventDestroy(e) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamDestroy(s1) == pcudaSuccess);
  BOOST_CHECK(pcudaStreamDestroy(s2) == pcudaSuccess);
}

BOOST_AUTO_TEST_SUITE_END()
