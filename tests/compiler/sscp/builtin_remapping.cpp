
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
#include <type_traits>
#include <sycl/sycl.hpp>
#include "common.hpp"

bool check_with_tolerance(double a, double b) {
  return std::abs(a - b) / std::abs(a) < 0.0001;
}


template<class T>
bool test() {
  sycl::queue q = get_queue();

  int num_functions = 12;

  T init = static_cast<T>(0.75);

  T* data = sycl::malloc_device<T>(num_functions, q);
  for(int i = 0; i < num_functions; ++i)
    q.memcpy(data + i, &init, sizeof(T));
  q.wait();

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
    T sin_value;
    T cos_value;
    if constexpr(std::is_same_v<T, float>)
      __builtin_sincosf(data[10], &sin_value, &cos_value);
    else
      __builtin_sincos(data[10], &sin_value, &cos_value);
    data[10] = sin_value;
    data[11] = cos_value;
  }).wait();

  std::vector<T> host(num_functions, T(0));
  q.memcpy(host.data(), data, sizeof(T) * num_functions).wait();

  // CHECK: 1
  std::cout << check_with_tolerance(host[0], std::sin(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[1], std::cos(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[2], std::pow(init, init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[3], std::pow(init, 3)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[4], std::exp(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[5], std::sqrt(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[6], std::tan(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[7], std::exp2(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[8], std::log(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[9], std::asin(init)) << std::endl;
  // CHECK: 1
  const bool sin_correct = check_with_tolerance(host[10], std::sin(init));
  std::cout << sin_correct << std::endl;
  // CHECK: 1
  const bool cos_correct = check_with_tolerance(host[11], std::cos(init));
  std::cout << cos_correct << std::endl;

  sycl::free(data, q);
  return sin_correct && cos_correct;
}

int main() {
  bool sincos_correct = test<float>();
  if(get_queue().get_device().has(sycl::aspect::fp64))
    sincos_correct &= test<double>();
  // CHECK: sincos checks: 1
  std::cout << "sincos checks: " << sincos_correct << std::endl;
  return sincos_correct ? 0 : 1;
}
