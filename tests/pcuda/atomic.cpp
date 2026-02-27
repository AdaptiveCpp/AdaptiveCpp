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

#include <pcuda.hpp>
#include <boost/mp11/list.hpp>
#include <boost/mp11/mpl.hpp>
#include <boost/test/unit_test.hpp>


BOOST_AUTO_TEST_SUITE(pcuda_atomic)

template<class T>
bool skip_test_for_type() {
  if constexpr (sizeof(T) > 4) {
    int dev;
    pcudaGetDevice(&dev);
    pcudaDeviceProp prop;
    pcudaGetDeviceProperties(&prop, dev);
    return !prop.pcudaHasAtomic64;
  }
  return false;
}

using add_test_types =
    boost::mp11::mp_list<int, unsigned int, unsigned long long, float, double>;

template<class T>
void test_atomic_add() {
  T* data;
  const size_t problem_size = 16;
  BOOST_CHECK(pcudaMallocManaged(&data, (problem_size + 1) * sizeof(T)) ==
              pcudaSuccess);
  for(int i = 0; i < problem_size+1; ++i)
    data[i] = T(i);


  pcudaParallelFor(1, problem_size, [=](){
    int gid = threadIdx.x;
    atomicAdd(data, data[gid+1]);
  });
  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);

  T reference = T(0);
  for(int i = 1; i < problem_size+1; ++i) {
    reference += T(i);
  }

  BOOST_CHECK(reference == *data);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

using sub_test_types =
    boost::mp11::mp_list<int, unsigned int>;

template<class T>
void test_atomic_sub() {
  T* data;
  const size_t problem_size = 16;
  const T offset = T(1024*1024);
  BOOST_CHECK(pcudaMallocManaged(&data, (problem_size + 1) * sizeof(T)) ==
              pcudaSuccess);
  for(int i = 0; i < problem_size+1; ++i)
    data[i] = T(i);
  data[0] = offset;

  pcudaParallelFor(1, problem_size, [=](){
    int gid = threadIdx.x;
    atomicSub(data, data[gid+1]);
  });
  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);

  T reference = offset;
  for(int i = 1; i < problem_size+1; ++i) {
    reference -= T(i);
  }

  BOOST_CHECK(reference == *data);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomicAdd, T, add_test_types) {
  if(skip_test_for_type<T>())
    return;
  test_atomic_add<T>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomicSub, T, sub_test_types) {
  if(skip_test_for_type<T>())
    return;
  test_atomic_sub<T>();
}


using min_test_types =
    boost::mp11::mp_list<int, unsigned int, unsigned long long, long long>;

template<class T>
void test_atomic_min() {
  // atomicMin/Max on ROCm seem to be bugged when managed memory is used.
  // Use device memory isntead.
  T* data;
  const size_t problem_size = 16;
  const T offset = T(1024*1024);
  BOOST_CHECK(pcudaMalloc(&data, (problem_size + 1) * sizeof(T)) ==
              pcudaSuccess);
  std::vector<T> input(problem_size+1);
  for(int i = 0; i < problem_size+1; ++i)
    input[i] = T(i);
  input[0] = offset;
  BOOST_CHECK(pcudaMemcpy(data, input.data(), input.size() * sizeof(T),
                          pcudaMemcpyDefault) == pcudaSuccess);

  pcudaParallelFor(1, problem_size, [=](){
    int gid = threadIdx.x;
    atomicMin(data, data[gid+1]);
  });
  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  BOOST_CHECK(pcudaMemcpy(input.data(), data, input.size() * sizeof(T),
                          pcudaMemcpyDefault) == pcudaSuccess);

  BOOST_CHECK(input[0] == 1);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(atomicMin, T, min_test_types) {
  if(skip_test_for_type<T>())
    return;
  test_atomic_min<T>();
}

using max_test_types =
    boost::mp11::mp_list<int, unsigned int, unsigned long long, long long>;

template<class T>
void test_atomic_max() {
  // atomicMin/Max on ROCm seem to be bugged when managed memory is used.
  // Use device memory isntead.
  T* data;
  const size_t problem_size = 16;
  const T offset = T(2);
  BOOST_CHECK(pcudaMalloc(&data, (problem_size + 1) * sizeof(T)) ==
              pcudaSuccess);
  std::vector<T> input(problem_size+1);
  for(int i = 0; i < problem_size+1; ++i)
    input[i] = T(i);
  input[0] = offset;
  BOOST_CHECK(pcudaMemcpy(data, input.data(), input.size() * sizeof(T),
                          pcudaMemcpyDefault) == pcudaSuccess);

  pcudaParallelFor(1, problem_size, [=](){
    int gid = threadIdx.x;
    atomicMax(data, data[gid+1]);
  });
  BOOST_CHECK(pcudaDeviceSynchronize() == pcudaSuccess);
  BOOST_CHECK(pcudaMemcpy(input.data(), data, input.size() * sizeof(T),
                          pcudaMemcpyDefault) == pcudaSuccess);
  BOOST_CHECK(input[0] == problem_size);

  BOOST_CHECK(pcudaFree(data) == pcudaSuccess);
}


BOOST_AUTO_TEST_CASE_TEMPLATE(atomicMax, T, max_test_types) {
  if(skip_test_for_type<T>())
    return;
  test_atomic_max<T>();
}

BOOST_AUTO_TEST_SUITE_END()
