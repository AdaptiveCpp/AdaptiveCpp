#pragma once

#include <array>
#include <sycl/sycl.hpp>

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

// Shared inline/template SYCL_KHR_FUNCTION in header (weak_odr linkage across TUs)
template <unsigned N>
SYCL_KHR_FUNCTION(sycl::khr::requires_work_group<N>)
inline int shared_header_helper(int x) {
  return x + static_cast<int>(N);
}

// Cross-TU free function kernels defined in tu1.cpp, launched in tu2.cpp
SYCL_KHR_KERNEL()
void cross_tu_small_struct(int* out, SmallStruct s);

SYCL_KHR_KERNEL()
void cross_tu_big_struct(int* out, BigStruct s);

SYCL_KHR_KERNEL()
void cross_tu_nested_struct(int* out, NestedStruct s);

// Cross-TU device helper with properties defined in tu1.cpp, called by kernel in tu2.cpp
SYCL_KHR_FUNCTION(
    sycl::khr::requires_work_group<64>,
    sycl::khr::sub_group_size<16>
)
int cross_tu_helper_with_wg64(int x);

void run_tu1_kernels(sycl::queue& q, int* d_out, int& host_res);
