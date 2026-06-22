// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

// UNSUPPORTED: vulkan || vk

#include "common.hpp"

void private_address_store_to_global(sycl::queue q) {
  // We just check the kernel compiles, there is no invalid
  // value that `*ptr` could contain.
  int **ptr = sycl::malloc_device<int *>(1, q);
  q.submit([&](sycl::handler &cgh) {
     cgh.single_task([=]() {
       int arr;
       *ptr = &arr;
     });
   }).wait();

  sycl::free(ptr, q);
}

void local_address_store_to_global(sycl::queue q) {
  // We just check the kernel compiles, there is no invalid
  // value that `*ptr` could contain.
  int **ptr = sycl::malloc_device<int *>(1, q);
  q.submit([&](sycl::handler &cgh) {
     auto scratch = sycl::local_accessor<int, 1>{1, cgh};
     cgh.single_task([=]() { *ptr = &scratch[0]; });
   }).wait();

  sycl::free(ptr, q);
}

void global_address_store_to_global(sycl::queue q) {
  int **ptr = sycl::malloc_device<int *>(1, q);
  int *target_ptr = sycl::malloc_device<int>(2, q);

  // CHECK: [[TARGET:0x[0-9a-fA-F]+]]
  std::cout << target_ptr + 1 << std::endl;

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task([=]() { *ptr = target_ptr + 1; });
  }).wait();

  int *host;
  q.memcpy(&host, ptr, sizeof(int *)).wait();

  // CHECK: [[TARGET]]
  std::cout << host << std::endl;

  sycl::free(ptr, q);
  sycl::free(target_ptr, q);
}

int main() {
  sycl::queue q = get_queue();

  private_address_store_to_global(q);
  local_address_store_to_global(q);
  global_address_store_to_global(q);
  return 0;
}
