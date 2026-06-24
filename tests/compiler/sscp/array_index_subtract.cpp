// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

// https://github.com/AdaptiveCpp/AdaptiveCpp/issues/2126
// UNSUPPORTED: vulkan || vk

#include "common.hpp"
#include <iostream>

int main() {
  sycl::queue q = get_queue();

  size_t size = 4;
  std::vector<int> in_data(size);
  for (int i = 0; i < size; ++i) {
    in_data[i] = i;
  }
  std::vector<int> out_data(size);

  int *in_ptr = sycl::malloc_device<int>(size, q);
  int *out_ptr = sycl::malloc_device<int>(size, q);

  q.memcpy(in_ptr, in_data.data(), size * sizeof(int)).wait();
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::range<1>(size),
        [=](sycl::id<1> id) {
          out_ptr[id] = in_ptr[size - 1];
        });
  }).wait();
  q.memcpy(out_data.data(), out_ptr, size * sizeof(int)).wait();

  // CHECK: 3
  // CHECK: 3
  // CHECK: 3
  // CHECK: 3
  for (int i = 0; i < size; ++i) {
    std::cout << out_data[i] << std::endl;
  }

  sycl::free(in_ptr, q);
  sycl::free(out_ptr, q);

  return 0;
}
