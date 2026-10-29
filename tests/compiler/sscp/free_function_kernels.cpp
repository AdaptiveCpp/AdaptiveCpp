// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include <iostream>
#include <array>
#include <sycl/sycl.hpp>
#include "common.hpp"

struct SmallStruct {
  int a;
  int b;
  int c;
};

struct BigStruct {
  int data[10];
};

struct NestedStruct {
  SmallStruct inner;
  std::array<int, 4> arr;
};

// Test basic free function kernel
SYCL_KHR_KERNEL()
void kernel_add(int* out, int a, int b) {
  *out = a + b;
}

// Test free function kernel with small aggregate passed by value (<=16 bytes)
SYCL_KHR_KERNEL()
void kernel_small_struct(int* out, SmallStruct s) {
  *out = s.a + s.b + s.c;
}

// Test free function kernel with big aggregate passed by value (>16 bytes)
SYCL_KHR_KERNEL()
void kernel_big_struct(int* out, BigStruct s) {
  int sum = 0;
  for (int i = 0; i < 10; ++i)
    sum += s.data[i];
  *out = sum;
}

// Test free function kernel with nested aggregate and std::array
SYCL_KHR_KERNEL()
void kernel_nested_struct(int* out, NestedStruct s) {
  int sum = s.inner.a + s.inner.b + s.inner.c;
  for (int i = 0; i < 4; ++i)
    sum += s.arr[i];
  *out = sum;
}

// Test SYCL_KHR_FUNCTION with properties
SYCL_KHR_FUNCTION(
    sycl::khr::requires_work_group<64>,
    sycl::khr::sub_group_size<32>
)
void kernel_with_properties(int* out, int val) {
  *out = val * 2;
}

// Test template-dependent property
template <unsigned N>
SYCL_KHR_FUNCTION(
    sycl::khr::requires_work_group<N>
)
void kernel_templated(int* out, int val) {
  *out = val + N;
}

int main() {
  sycl::queue q = get_queue();

  int* d_out = sycl::malloc_device<int>(1, q);
  int host_res = 0;

  // 1. Basic launch_task
  sycl::khr::launch_task(q, sycl::khr::kernel_function<kernel_add>, d_out, 15, 27);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 42
  std::cout << host_res << std::endl;

  // 2. Small struct
  SmallStruct s1{10, 20, 12};
  sycl::khr::launch_task(q, sycl::khr::kernel_function<kernel_small_struct>, d_out, s1);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 42
  std::cout << host_res << std::endl;

  // 3. Big struct
  BigStruct s2{};
  for (int i = 0; i < 10; ++i)
    s2.data[i] = i + 1; // sum is 55
  sycl::khr::launch_task(q, sycl::khr::kernel_function<kernel_big_struct>, d_out, s2);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 55
  std::cout << host_res << std::endl;

  // 4. Nested struct
  NestedStruct s3{{1, 2, 3}, {4, 5, 6, 7}}; // sum is 28
  sycl::khr::launch_task(q, sycl::khr::kernel_function<kernel_nested_struct>, d_out, s3);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 28
  std::cout << host_res << std::endl;

  // 5. launch_grouped with properties
  sycl::khr::launch_grouped(q, sycl::range<1>{64}, sycl::range<1>{64},
                            sycl::khr::kernel_function<kernel_with_properties>, d_out, 21);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 42
  std::cout << host_res << std::endl;

  // 6. Templated kernel with property
  sycl::khr::launch_grouped(q, sycl::range<1>{128}, sycl::range<1>{128},
                            sycl::khr::kernel_function<kernel_templated<128>>, d_out, 10);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 138
  std::cout << host_res << std::endl;

  sycl::free(d_out, q);
  return 0;
}
