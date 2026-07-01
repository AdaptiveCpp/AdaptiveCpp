// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include "common.hpp"
#include <iostream>

// Test compiler can handle lowering llvm.memset intrinsics with both
// a compile time known length parameter and a runtime length parameter.

void dynamic_size(sycl::queue q) {
  size_t size = 4; // Passed as argument rather than literal constant
  int *device_ptr = sycl::malloc_device<int>(size, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       for (size_t i = 0; i < size; i++)
         device_ptr[i] = -1;
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), device_ptr, size * sizeof(int)).wait();

  // CHECK: -1
  // CHECK: -1
  // CHECK: -1
  // CHECK: -1
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  sycl::free(device_ptr, q);
}

void static_size(sycl::queue q) {
  const int size = 4; // Folded into device code as literal constant
  int *device_ptr = sycl::malloc_device<int>(size, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       for (size_t i = 0; i < size; i++)
         device_ptr[i] = -1;
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), device_ptr, size * sizeof(int)).wait();

  // CHECK: -1
  // CHECK: -1
  // CHECK: -1
  // CHECK: -1
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  sycl::free(device_ptr, q);
}
int main() {
  sycl::queue q = get_queue();
  dynamic_size(q);
  static_size(q);
  return 0;
}
