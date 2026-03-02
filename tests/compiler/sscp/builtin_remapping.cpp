
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <cmath>
#include <sycl/sycl.hpp>
#include "common.hpp"

bool check_with_tolerance(double a, double b) {
  return std::abs(a - b) / std::abs(a) < 0.0001;
}


template<class T>
void test() {
  sycl::queue q = get_queue();

  int num_functions = 10;

  T init = static_cast<T>(0.75);

  T* data = sycl::malloc_shared<T>(num_functions, q);
  for(int i = 0; i < num_functions; ++i)
    data[i] = init;

  q.single_task([=](){
    data[0] = std::sin(data[0]);
    data[1] = std::cos(data[1]);
    data[2] = std::pow(data[2], init);
    // pow(float, int) is missing in standard after C++11,
    // will be replaced to llvm.pow.f32.i32 for -O3 -ffast-math
    data[3] = std::pow(data[3], T(3));
    data[4] = std::exp(data[4]);
    data[5] = std::sqrt(data[5]);
    data[6] = std::tan(data[6]);
    data[7] = std::exp2(data[7]);
    data[8] = std::log(data[8]);
    data[9] = std::asin(data[9]);
  }).wait();

  // CHECK: 1
  std::cout << check_with_tolerance(data[0], std::sin(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[1], std::cos(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[2], std::pow(init, init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[3], std::pow(init, 3)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[4], std::exp(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[5], std::sqrt(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[6], std::tan(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[7], std::exp2(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[8], std::log(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(data[9], std::asin(init)) << std::endl;

  sycl::free(data, q);
}

int main() {
  test<float>();
  if(get_queue().get_device().has(sycl::aspect::fp64))
    test<double>();
}
