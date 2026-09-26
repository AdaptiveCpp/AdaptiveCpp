// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

// Non-power-of-two integers need legalization on some backends (e.g. Metal)
// Compare device and host results for the same code
// On Metal i96 is promoted to i128, which only supports bitwise operations,
// shifts by multiples of 32 and comparisons, so the i96 cases stick to those

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

using i48 = _BitInt(48);
using u48 = unsigned _BitInt(48);
using i96 = _BitInt(96);
using u96 = unsigned _BitInt(96);
using u128 = unsigned __int128;
using i128 = __int128;

constexpr int NumOut = 32;
constexpr int NumMem = 8;

struct Inputs {
  u96 a;       // uses the upper 32 bits
  u96 b;       // small, nonzero
  i96 n;       // negative
  i96 m;       // positive
  int32_t s32; // negative
  i48 s48;     // negative
  u48 z48;     // uses the upper bits
  int cond;
  int trip;
};

struct Out {
  uint64_t *o;
  int k = 0;
  void put(u128 v) {
    o[k++] = static_cast<uint64_t>(v);
    o[k++] = static_cast<uint64_t>(v >> 64);
  }
  void put(u96 v) { put(static_cast<u128>(v)); }
  void put(i96 v) { put(static_cast<u128>(static_cast<u96>(v))); }
};

template <class F>
bool run_case(sycl::queue &q, const Inputs &host_in, F f) {
  Inputs *in = sycl::malloc_shared<Inputs>(1, q);
  uint32_t *mem = sycl::malloc_shared<uint32_t>(NumMem, q);
  uint64_t *out = sycl::malloc_shared<uint64_t>(NumOut, q);

  uint32_t ref_mem[NumMem];
  uint64_t ref_out[NumOut] = {};
  for (int i = 0; i < NumMem; ++i)
    mem[i] = ref_mem[i] = 0xA5A5A5A5u + i;
  for (int i = 0; i < NumOut; ++i)
    out[i] = 0;
  *in = host_in;

  q.parallel_for(sycl::range<1>{1}, [=](sycl::id<1>) {
    f(in, mem, out);
  }).wait();

  Inputs ref_in = host_in;
  f(&ref_in, ref_mem, ref_out);

  bool ok = std::memcmp(out, ref_out, sizeof(ref_out)) == 0 &&
            std::memcmp(mem, ref_mem, sizeof(ref_mem)) == 0;

  sycl::free(in, q);
  sycl::free(mem, q);
  sycl::free(out, q);
  return ok;
}

// Avoid short-circuiting: LLVM can merge the three component comparisons
// into one i96 comparison
inline bool vec3_equals(sycl::vec<uint32_t, 3> a, sycl::vec<uint32_t, 3> b) {
  bool eqx = a.x() == b.x();
  bool eqy = a.y() == b.y();
  bool eqz = a.z() == b.z();
  return eqx && eqy && eqz;
}

bool test_vec3_equals(sycl::queue &q) {
  using u32_3 = sycl::vec<uint32_t, 3>;
  constexpr size_t n = 4;
  u32_3 *a = sycl::malloc_shared<u32_3>(n, q);
  u32_3 *b = sycl::malloc_shared<u32_3>(n, q);
  uint8_t *r = sycl::malloc_shared<uint8_t>(n, q);
  for (size_t i = 0; i < n; ++i) {
    a[i] = u32_3{1u, 2u, static_cast<uint32_t>(i)};
    b[i] = u32_3{1u, 2u, 3u};
  }
  q.parallel_for(sycl::range<1>{n}, [=](sycl::item<1> it) {
    size_t i = it.get_linear_id();
    r[i] = vec3_equals(a[i], b[i]);
  }).wait();
  bool ok = r[0] == 0 && r[1] == 0 && r[2] == 0 && r[3] == 1;
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(r, q);
  return ok;
}

int main() {
  sycl::queue q = get_queue();

  Inputs in{};
  in.a = (static_cast<u96>(0xF1234567u) << 64) | 0x89ABCDEF01234567ull;
  in.b = 7;
  in.n = -(static_cast<i96>(0x1234567u) << 64) - 12345;
  in.m = 5;
  in.s32 = -42;
  in.s48 = -1000000;
  in.z48 = static_cast<u48>(0xFEDCBA987654ull);
  in.cond = 1;
  in.trip = 10;

  // CHECK: vec3_equals: 1
  std::cout << "vec3_equals: " << test_vec3_equals(q) << std::endl;

  // CHECK: bitwise: 1
  std::cout << "bitwise: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    Out out{o};
    u96 x = in->a, y = in->b;
    out.put(x & y);
    out.put(x | y);
    out.put(x ^ y);
    out.put(~x);
  }) << std::endl;

  // CHECK: shift: 1
  std::cout << "shift: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    Out out{o};
    u96 x = in->a;
    i96 n = in->n;
    out.put(x << 32);
    out.put(x << 64);
    out.put(x >> 32);
    out.put(x >> 64);
    out.put(n >> 32);
    out.put(n >> 64);
  }) << std::endl;

  // CHECK: compare: 1
  std::cout << "compare: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    i96 x = in->n, y = in->m;
    o[0] = x < y;
    o[1] = x >= y;
    o[2] = x > y;
    o[3] = x <= y;
    o[4] = static_cast<u96>(x) < static_cast<u96>(y);
    o[5] = static_cast<u96>(x) >= static_cast<u96>(y);
    o[6] = in->a > in->b;
    o[7] = in->a <= in->b;
    o[8] = x == y;
    o[9] = x != y;
  }) << std::endl;

  // CHECK: extend: 1
  std::cout << "extend: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    Out out{o};
    out.put(static_cast<i96>(in->s32));                   // sext i32 -> i96
    out.put(static_cast<u128>(static_cast<i128>(in->n))); // sext i96 -> i128
    out.put(static_cast<u128>(in->a));                    // zext i96 -> i128
    out.put(static_cast<i96>(in->s48));                   // sext i48 -> i96
    out.put(static_cast<u96>(in->z48));                   // zext i48 -> i96
    o[out.k++] = static_cast<uint64_t>(static_cast<int64_t>(in->s48)); // sext i48 -> i64
    o[out.k++] = static_cast<uint64_t>(static_cast<i48>(in->n));       // trunc i96 -> i48
  }) << std::endl;

  // CHECK: select: 1
  std::cout << "select: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    Out out{o};
    i96 x = in->n ^ in->m, y = in->n | in->m;
    out.put(in->cond ? x : y);
    out.put(!in->cond ? x : y);
  }) << std::endl;

  // CHECK: loop: 1
  std::cout << "loop: " << run_case(q, in, [](const Inputs *in, uint32_t *, uint64_t *o) {
    Out out{o};
    u96 acc = in->a;
    for (int i = 0; i < in->trip; ++i) {
      acc = (acc << 32) ^ (acc >> 64) ^ static_cast<u96>(i);
    }
    out.put(acc);
  }) << std::endl;

  // An aligned i96 load/store must not touch mem[3].
  // CHECK: load_store_i96: 1
  std::cout << "load_store_i96: " << run_case(q, in, [](const Inputs *in, uint32_t *mem, uint64_t *o) {
    u96 x = 0;
    __builtin_memcpy(&x, mem, 12);
    x = ~x ^ in->b;
    __builtin_memcpy(mem, &x, 12);
    Out{o}.put(x);
  }) << std::endl;

  // An aligned i48 load/store must preserve the upper half of mem[1].
  // CHECK: load_store_i48: 1
  std::cout << "load_store_i48: " << run_case(q, in, [](const Inputs *in, uint32_t *mem, uint64_t *o) {
    u48 x = 0;
    __builtin_memcpy(&x, mem, 6);
    x = x * 3 + in->z48;
    __builtin_memcpy(mem, &x, 6);
    o[0] = static_cast<uint64_t>(x);
  }) << std::endl;
}
