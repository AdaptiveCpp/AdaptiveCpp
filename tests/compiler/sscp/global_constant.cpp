// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <array>

#include <sycl/sycl.hpp>
#include "common.hpp"

constexpr int table [] = {10, 20, 30, 40};
constexpr std::array<int, 4> array_table = {100, 200, 300, 400};

void test_c_array(sycl::queue& q) {
  int* data = sycl::malloc_device<int>(1024, q);
  q.parallel_for(sycl::range{1024}, [=](auto idx){
    data[idx] = idx + table[idx % 4];
  }).wait();

  std::vector<int> result(1024);
  q.memcpy(result.data(), data, result.size()).wait();
  for(int i = 0; i < 8; ++i) {
    // CHECK: 10
    // CHECK: 21
    // CHECK: 32
    // CHECK: 43
    // CHECK: 14
    // CHECK: 25
    // CHECK: 36
    // CHECK: 47
    std::cout << result[i] << std::endl;
  }
  sycl::free(data, q);
}

void test_std_array(sycl::queue& q) {
  int* data = sycl::malloc_device<int>(4, q);
  q.single_task([=]() {
    for (int i = 0; i < 4; ++i)
      data[i] = array_table[i];
  }).wait();

  std::vector<int> result(4);
  q.memcpy(result.data(), data, sizeof(int) * 4).wait();
  for (int i = 0; i < 4; ++i) {
    // CHECK: 100
    // CHECK: 200
    // CHECK: 300
    // CHECK: 400
    std::cout << result[i] << std::endl;
  }
  sycl::free(data, q);
}

int main() {
  sycl::queue q = get_queue();
  test_c_array(q);
  test_std_array(q);
}
