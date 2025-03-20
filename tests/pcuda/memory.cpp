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


BOOST_AUTO_TEST_SUITE(pcuda_memory);


BOOST_AUTO_TEST_CASE(MallocManaged) {
  int* data;
  int problem_size = 1024;
  int group_size = 128;
  BOOST_TEST(pcudaMallocManaged(&data, problem_size * sizeof(int)) == pcudaSuccess);

  for(int i = 0; i < problem_size; ++i)
    data[i] = i;

  auto err = pcudaParallelFor(problem_size / group_size, group_size, [=](){
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    data[gid] +=1;
  });
  BOOST_CHECK(err == pcudaSuccess);

  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 1);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(MallocHost) {
  int* data;
  int problem_size = 1024;
  int group_size = 128;
  BOOST_CHECK(pcudaMallocHost(&data, problem_size * sizeof(int)) == pcudaSuccess);

  for(int i = 0; i < problem_size; ++i)
    data[i] = i;

  auto err = pcudaParallelFor(problem_size / group_size, group_size, [=](){
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    data[gid] +=1;
  });
  BOOST_CHECK(err == pcudaSuccess);

  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data[i] == i + 1);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(Malloc) {
  int* data;
  int* data_in;
  int* data_out;
  int problem_size = 1024;
  int group_size = 128;
  BOOST_CHECK(pcudaMalloc(&data, problem_size * sizeof(int)) == pcudaSuccess);
  BOOST_CHECK(pcudaMallocManaged(&data_in, problem_size * sizeof(int)) == pcudaSuccess);
  BOOST_CHECK(pcudaMallocManaged(&data_out, problem_size * sizeof(int)) == pcudaSuccess);

  
  for(int i = 0; i < problem_size; ++i)
    data_in[i] = i;

  pcudaParallelFor(problem_size / group_size, group_size, [=](){
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    data[gid] = data_in[gid] + 1;
  });

  pcudaParallelFor(problem_size / group_size, group_size, [=](){
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    data_out[gid] = data[gid];
  });

  pcudaDeviceSynchronize();

  for(int i = 0; i < problem_size; ++i) {
    BOOST_CHECK(data_out[i] == i + 1);
  }

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data_in) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data_out) == pcudaSuccess);
}


BOOST_AUTO_TEST_CASE(Memcpy) {
  int* data1;
  int* data2;
  int problem_size = 1024;
  int group_size = 128;
  std::vector<int> host_input(problem_size);
  std::vector<int> host_output(problem_size);

  for(int i = 0; i < host_input.size(); ++i)
    host_input[i] = i;

  BOOST_CHECK(pcudaMalloc(&data1, problem_size * sizeof(int)) == pcudaSuccess);
  BOOST_CHECK(pcudaMalloc(&data2, problem_size * sizeof(int)) == pcudaSuccess);

  // h2d
  BOOST_CHECK(pcudaMemcpy(data1, host_input.data(), problem_size * sizeof(int),
                          pcudaMemcpyHostToDevice) == pcudaSuccess);

  // d2d
  BOOST_CHECK(pcudaMemcpy(data2, data1, problem_size * sizeof(int),
                          pcudaMemcpyDeviceToDevice) == pcudaSuccess);

  // d2h
  BOOST_CHECK(pcudaMemcpy(host_output.data(), data2, problem_size * sizeof(int),
                          pcudaMemcpyDeviceToHost) == pcudaSuccess);

  BOOST_CHECK(host_input == host_output);

  BOOST_CHECK(pcudaFree(data1) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data2) == pcudaSuccess);
}


BOOST_AUTO_TEST_CASE(MemcpyAsync) {
  int* data1;
  int* data2;
  int problem_size = 1024;
  int group_size = 128;
  std::vector<int> host_input(problem_size);
  std::vector<int> host_output(problem_size);

  for(int i = 0; i < host_input.size(); ++i)
    host_input[i] = i;

  BOOST_CHECK(pcudaMalloc(&data1, problem_size * sizeof(int)) == pcudaSuccess);
  BOOST_CHECK(pcudaMalloc(&data2, problem_size * sizeof(int)) == pcudaSuccess);

  // h2d
  BOOST_CHECK(pcudaMemcpyAsync(data1, host_input.data(),
                               problem_size * sizeof(int),
                               pcudaMemcpyHostToDevice, 0) == pcudaSuccess);

  // d2d
  BOOST_CHECK(pcudaMemcpyAsync(data2, data1, problem_size * sizeof(int),
                               pcudaMemcpyDeviceToDevice, 0) == pcudaSuccess);

  // d2h
  BOOST_CHECK(pcudaMemcpyAsync(host_output.data(), data2,
                               problem_size * sizeof(int),
                               pcudaMemcpyDeviceToHost, 0) == pcudaSuccess);

  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  BOOST_CHECK(host_input == host_output);

  BOOST_CHECK(pcudaFree(data1) == pcudaSuccess);
  BOOST_CHECK(pcudaFree(data2) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE(Memset) {
  int* data;
  int problem_size = 1024;
  BOOST_CHECK(pcudaMallocManaged(&data, problem_size * sizeof(int)) ==
              pcudaSuccess);
  BOOST_CHECK(pcudaMemset(data, 42, problem_size * sizeof(int)) == pcudaSuccess);

  char* char_data = reinterpret_cast<char*>(data);
  for(int i = 0; i < problem_size * sizeof(int); ++i)
    BOOST_CHECK(char_data[i] == 42);

  BOOST_CHECK(pcudaMemsetAsync(data, 44, problem_size * sizeof(int), 0) ==
              pcudaSuccess);
  BOOST_CHECK(pcudaStreamSynchronize(0) == pcudaSuccess);
  for(int i = 0; i < problem_size * sizeof(int); ++i)
    BOOST_CHECK(char_data[i] == 44);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_SUITE_END()
