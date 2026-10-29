#include "header.hpp"

// Internal linkage (static) SYCL_KHR_FUNCTION with the same name as in tu2.cpp
SYCL_KHR_FUNCTION()
static int internal_helper(int x) {
  return x + 10;
}

SYCL_KHR_KERNEL()
void cross_tu_small_struct(int* out, SmallStruct s) {
  *out = s.a + s.b + s.c;
}

SYCL_KHR_KERNEL()
void cross_tu_big_struct(int* out, BigStruct s) {
  int sum = 0;
  for (int i = 0; i < 10; ++i)
    sum += s.data[i];
  *out = sum;
}

SYCL_KHR_KERNEL()
void cross_tu_nested_struct(int* out, NestedStruct s) {
  int sum = s.inner.a + s.inner.b + s.inner.c;
  for (int i = 0; i < 4; ++i)
    sum += s.arr[i];
  *out = sum;
}

SYCL_KHR_FUNCTION(
    sycl::khr::requires_work_group<64>,
    sycl::khr::sub_group_size<16>
)
int cross_tu_helper_with_wg64(int x) {
  return internal_helper(x) + shared_header_helper<64>(0);
}

SYCL_KHR_FUNCTION(sycl::khr::requires_work_group<64>)
void tu1_kernel(int* out, int x) {
  *out = internal_helper(x) + shared_header_helper<64>(0);
}

void run_tu1_kernels(sycl::queue& q, int* d_out, int& host_res) {
  sycl::khr::launch_grouped(q, sycl::range<1>{64}, sycl::range<1>{64},
                            sycl::khr::kernel_function<tu1_kernel>, d_out, 5);
  q.wait();
  q.memcpy(&host_res, d_out, sizeof(int)).wait();
}
