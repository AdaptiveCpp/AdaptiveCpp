// RUN: %acpp %s %S/tu1.cpp -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s %S/tu1.cpp -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include <iostream>
#include "header.hpp"
#include "../common.hpp"

// Internal linkage (static) SYCL_KHR_FUNCTION with the same name as in tu1.cpp
SYCL_KHR_FUNCTION()
static int internal_helper(int x) {
  return x + 100;
}

// Kernel in tu2.cpp calling cross-TU helper from tu1.cpp (which has requires_work_group<64>
// and sub_group_size<16>), shared_header_helper<64> (weak_odr in both TUs), and tu2's static
// internal_helper.
SYCL_KHR_KERNEL()
void tu2_kernel_calling_tu1_helper(int* out, int x) {
  // cross_tu_helper_with_wg64(x) = (x + 10) + 64
  // internal_helper(0) in tu2 = 100
  // shared_header_helper<64>(0) = 64
  *out = cross_tu_helper_with_wg64(x) + internal_helper(0) + shared_header_helper<64>(0);
}

int main() {
  sycl::queue q = get_queue();
  int* d_out = sycl::malloc_device<int>(1, q);
  int host_res = 0;

  // 1. Cross-TU small struct by-value
  SmallStruct s1{10, 20, 12};
  sycl::khr::launch_task(q, sycl::khr::kernel_function<cross_tu_small_struct>, d_out, s1);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 42
  std::cout << host_res << std::endl;

  // 2. Cross-TU big struct by-value (>16 bytes, byval pointer in Clang x86-64 ABI)
  BigStruct s2{};
  for (int i = 0; i < 10; ++i)
    s2.data[i] = i + 1;
  sycl::khr::launch_task(q, sycl::khr::kernel_function<cross_tu_big_struct>, d_out, s2);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 55
  std::cout << host_res << std::endl;

  // 3. Cross-TU nested struct with std::array by-value
  NestedStruct s3{{1, 2, 3}, {4, 5, 6, 7}};
  sycl::khr::launch_task(q, sycl::khr::kernel_function<cross_tu_nested_struct>, d_out, s3);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 28
  std::cout << host_res << std::endl;

  // 3b. Cross-TU std::tuple by-value (lowered as bare ptr without byval on x86-64 Linux)
  std::tuple<int, int> t1{19, 23};
  sycl::khr::launch_task(q, sycl::khr::kernel_function<cross_tu_tuple>, d_out, t1);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 42
  std::cout << host_res << std::endl;

  // 4. Kernel launched from tu1.cpp using tu1's static internal_helper + shared_header_helper<64>
  // (5 + 10) + 64 = 79
  run_tu1_kernels(q, d_out, host_res);
  // CHECK: 79
  std::cout << host_res << std::endl;

  // 5. Kernel in tu2.cpp calling cross_tu_helper_with_wg64 in tu1.cpp, shared_header_helper<64>,
  // and tu2's static internal_helper:
  // ((2 + 10) + 64) + 100 + 64 = 240
  sycl::khr::launch_grouped(q, sycl::range<1>{64}, sycl::range<1>{64},
                            sycl::khr::kernel_function<tu2_kernel_calling_tu1_helper>, d_out, 2);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
  // CHECK: 240
  std::cout << host_res << std::endl;

  sycl::free(d_out, q);
  return 0;
}
