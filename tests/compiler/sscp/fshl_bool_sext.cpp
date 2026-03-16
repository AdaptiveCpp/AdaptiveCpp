// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

// likely expanded to @llvm.fshl.i32
inline uint32_t rotate_left(uint32_t x, uint32_t n) {
  return (x << n) | (x >> (32u - n));
}

// likely expanded to @llvm.fshr.i32
inline uint32_t rotate_right(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32u - n));
}

// likely expanded to @llvm.fshl.i32
inline uint32_t funnel_left(uint32_t a, uint32_t b, uint32_t s) {
  return (a << s) | (b >> (32u - s));
}

// likely expanded to @llvm.fshr.i32
inline uint32_t funnel_right(uint32_t a, uint32_t b, uint32_t s) {
  return (b >> s) | (a << (32u - s));
}

// likely expanded to sext i1 to i32
inline int32_t bool_sign_mask(int32_t a, int32_t b) {
  return -(int32_t)(a > b);
}

int main() {
  sycl::queue q = get_queue();

  uint32_t *data = sycl::malloc_shared<uint32_t>(8, q);
  int32_t *sdata = sycl::malloc_shared<int32_t>(4, q);

  data[0] = 0xDEADBEEFu;
  data[1] = 12;
  data[2] = 0xAAAAAAAAu;
  data[3] = 0x55555555u;
  data[4] = 8;

  sdata[0] = 42;
  sdata[1] = 10;
  sdata[2] = 5;
  sdata[3] = 100;

  q.single_task([=]() {
    uint32_t x = data[0];
    uint32_t n = data[1];
    uint32_t a = data[2];
    uint32_t b = data[3];
    uint32_t s = data[4];

    data[5] = rotate_left(x, n);
    data[6] = rotate_right(x, n);
    data[7] = funnel_left(a, b, s);

    sdata[2] = bool_sign_mask(sdata[0], sdata[1]);
    sdata[3] = bool_sign_mask(sdata[1], sdata[0]);
  }).wait();

  uint32_t expected_rotl = rotate_left(0xDEADBEEFu, 12);
  uint32_t expected_rotr = rotate_right(0xDEADBEEFu, 12);
  uint32_t expected_fshl = funnel_left(0xAAAAAAAAu, 0x55555555u, 8);

  // CHECK: rotl: 1
  std::cout << "rotl: " << (data[5] == expected_rotl) << std::endl;
  // CHECK: rotr: 1
  std::cout << "rotr: " << (data[6] == expected_rotr) << std::endl;
  // CHECK: fshl: 1
  std::cout << "fshl: " << (data[7] == expected_fshl) << std::endl;
  // CHECK: sext_true: -1
  std::cout << "sext_true: " << sdata[2] << std::endl;
  // CHECK: sext_false: 0
  std::cout << "sext_false: " << sdata[3] << std::endl;

  sycl::free(data, q);
  sycl::free(sdata, q);
}
