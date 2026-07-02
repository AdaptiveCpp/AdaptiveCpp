// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

// Fails on CI with AMD EPYC 7763 OpenCL driver
// UNSUPPORTED: opencl || ocl

#include "common.hpp"
#include <iostream>

// Test compiler can handle lowering llvm.memmove intrinsics with both
// a compile time known length parameter and a runtime length parameter.

void dynamic_size_backward(sycl::queue q) {
  int size = 4;
  int init[4] = {0, 1, 2, 3};

  // Test global address space memmove
  int *device_ptr = sycl::malloc_device<int>(size, q);
  q.memcpy(device_ptr, init, sizeof(init)).wait();

  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       for (size_t i = 0; i < size - 1; i++) {
         device_ptr[i] = device_ptr[i + 1];
       }
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), device_ptr, size * sizeof(int)).wait();

  // CHECK: 1
  // CHECK: 2
  // CHECK: 3
  // CHECK: 3
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  std::cout << std::endl;
  sycl::free(device_ptr, q);
}

void dynamic_size_forward(sycl::queue q) {
  int size = 4;
  int init[4] = {0, 1, 2, 3};

  int *device_ptr = sycl::malloc_device<int>(size, q);
  q.memcpy(device_ptr, init, sizeof(init)).wait();

  // Test global address space memmove
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       for (size_t i = size - 1; i > 0; i--) {
         device_ptr[i] = device_ptr[i - 1];
       }
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), device_ptr, size * sizeof(int)).wait();

  // CHECK: 0
  // CHECK: 0
  // CHECK: 1
  // CHECK: 2
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  std::cout << std::endl;

  sycl::free(device_ptr, q);
}

void static_size_backward(sycl::queue q) {
  constexpr size_t size = 4;
  int init[4] = {0, 1, 2, 3};

  int *input = sycl::malloc_device<int>(size, q);
  int *output = sycl::malloc_device<int>(size, q);
  q.memcpy(input, init, sizeof(init)).wait();

  // Test private address space memmove
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       int arr[size];
       for (size_t i = 0; i < size; i++) {
         arr[i] = input[i];
       }

       for (size_t i = 0; i < size - 1; i++) {
         arr[i] = arr[i + 1];
       }

       for (size_t i = 0; i < size; i++) {
         output[i] = arr[i];
       }
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), output, size * sizeof(int)).wait();

  // CHECK: 1
  // CHECK: 2
  // CHECK: 3
  // CHECK: 3
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  std::cout << std::endl;

  sycl::free(input, q);
  sycl::free(output, q);
}

void static_size_forward(sycl::queue q) {
  constexpr size_t size = 4;
  int init[4] = {0, 1, 2, 3};

  int *input = sycl::malloc_device<int>(size, q);
  int *output = sycl::malloc_device<int>(size, q);
  q.memcpy(input, init, sizeof(init)).wait();

  // Test private address space memmove
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       float arr[size];
       for (size_t i = 0; i < size; i++) {
         arr[i] = input[i];
       }

       for (size_t i = size - 1; i > 0; i--) {
         arr[i] = arr[i - 1];
       }

       for (size_t i = 0; i < size; i++) {
         output[i] = arr[i];
       }
     });
   }).wait();

  std::vector<int> out_data(size);
  q.memcpy(out_data.data(), output, size * sizeof(int)).wait();

  // CHECK: 0
  // CHECK: 0
  // CHECK: 1
  // CHECK: 2
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }
  std::cout << std::endl;

  sycl::free(input, q);
  sycl::free(output, q);
}

int main() {
  sycl::queue q = get_queue();
  dynamic_size_backward(q);
  dynamic_size_forward(q);
  static_size_backward(q);
  static_size_forward(q);
  return 0;
}
