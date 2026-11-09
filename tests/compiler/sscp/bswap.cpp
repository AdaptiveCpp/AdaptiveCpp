// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include <cstdint>
#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

// @llvm.bswap.i16, @llvm.bswap.i32 and @llvm.bswap.i64
inline uint16_t bswap16(uint16_t x) { return __builtin_bswap16(x); }
inline uint32_t bswap32(uint32_t x) { return __builtin_bswap32(x); }
inline uint64_t bswap64(uint64_t x) { return __builtin_bswap64(x); }

int main() {
  sycl::queue q = get_queue();

  uint16_t *data16 = sycl::malloc_device<uint16_t>(2, q);
  uint32_t *data32 = sycl::malloc_device<uint32_t>(2, q);
  uint64_t *data64 = sycl::malloc_device<uint64_t>(2, q);

  std::vector<uint16_t> host_data16 {0x1122u, 0};
  std::vector<uint32_t> host_data32 {0x11223344u, 0};
  std::vector<uint64_t> host_data64 {0x1122334455667788ull, 0};

  q.memcpy(data16, host_data16.data(), 2 * sizeof(uint16_t));
  q.memcpy(data32, host_data32.data(), 2 * sizeof(uint32_t));
  q.memcpy(data64, host_data64.data(), 2 * sizeof(uint64_t));
  q.wait();

  q.single_task([=]() {
    data16[1] = bswap16(data16[0]);
    data32[1] = bswap32(data32[0]);
    data64[1] = bswap64(data64[0]);
  }).wait();

  q.memcpy(host_data16.data(), data16, 2 * sizeof(uint16_t));
  q.memcpy(host_data32.data(), data32, 2 * sizeof(uint32_t));
  q.memcpy(host_data64.data(), data64, 2 * sizeof(uint64_t));
  q.wait();

  // CHECK: bswap16: 1
  std::cout << "bswap16: " << (host_data16[1] == 0x2211u) << std::endl;
  // CHECK: bswap32: 1
  std::cout << "bswap32: " << (host_data32[1] == 0x44332211u) << std::endl;
  // CHECK: bswap64: 1
  std::cout << "bswap64: " << (host_data64[1] == 0x8877665544332211ull) << std::endl;

  sycl::free(data16, q);
  sycl::free(data32, q);
  sycl::free(data64, q);
}
