
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -fno-math-errno
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

  int num_functions = 21;

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
    // float builtins with an 'f' in the base name (erff, fdimf, ...)
    data[10] = std::erf(data[10]);
    data[11] = std::erfc(data[11]);
    data[12] = std::fdim(data[12], T(0.5));
    data[13] = std::fmod(data[13], T(0.5));
    data[14] = std::nextafter(data[14], T(1));
    data[15] = std::sinh(data[15]);
    data[16] = std::cosh(data[16]);
    data[17] = std::tanh(data[17]);
    data[18] = std::atan2(data[18], T(0.5));
    // sycl::tan declares the builtin that replaces llvm.tan from std::tan
    data[19] = sycl::tan(data[19]);
    // llvm.modf returns both parts; their sum must give back the argument
    T integral_part;
    data[20] = std::modf(data[20] * T(3), &integral_part) + integral_part;
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
  std::cout << check_with_tolerance(host[10], std::erf(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[11], std::erfc(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[12], std::fdim(init, T(0.5))) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[13], std::fmod(init, T(0.5))) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[14], std::nextafter(init, T(1))) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[15], std::sinh(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[16], std::cosh(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[17], std::tanh(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[18], std::atan2(init, T(0.5))) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[19], std::tan(init)) << std::endl;
  // CHECK: 1
  std::cout << check_with_tolerance(host[20], init * T(3)) << std::endl;

  sycl::free(data, q);
}

int main() {
  test<float>();
  if(get_queue().get_device().has(sycl::aspect::fp64))
    test<double>();
}
