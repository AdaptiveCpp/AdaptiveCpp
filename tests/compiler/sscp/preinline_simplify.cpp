// Functions are simplified before being inlined into the kernel, so the
// `a && b && c` below must reach the JIT pipeline already folded into selects.
//
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -f %t.ll
// RUN: ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS=%t.ll %t | FileCheck %s --check-prefix=OUT
// RUN: FileCheck %s --input-file %t.ll

#include "common.hpp"
#include <iostream>

struct bounds {
  int lo;
  int hi;
  int skip;
};

// A real function, so that its reference parameter carries `dereferenceable`.
static int classify(const bounds &b, int x, int y, int z) {
  if (x > b.lo && y > b.lo && z > b.lo)
    return 1;
  return 0;
}

int main() {
  sycl::queue q = get_queue();

  constexpr size_t size = 8;
  bounds *b = sycl::malloc_device<bounds>(1, q);
  int *out = sycl::malloc_device<int>(size, q);
  const bounds host_b{2, 7, 4};
  q.memcpy(b, &host_b, sizeof(bounds)).wait();

  q.parallel_for(sycl::range<1>(size), [=](sycl::id<1> id) {
     int x = static_cast<int>(id[0]);
     int y = 7 - x;
     int z = 2 * x;
     out[id] = classify(*b, x, y, z);
   }).wait();

  std::vector<int> result(size);
  q.memcpy(result.data(), out, size * sizeof(int)).wait();

  // OUT: 0 0 0 1 1 0 0 0
  for (size_t i = 0; i < size; ++i)
    std::cout << result[i] << (i + 1 < size ? " " : "\n");

  sycl::free(b, q);
  sycl::free(out, q);
  return 0;
}

// No short-circuit block of `classify` may survive right after inlining; the
// kernel keeps other branches of its own, hence the check on block names.
//
// CHECK-LABEL: stage: jit_optimizations
// CHECK-LABEL: define {{.*}}__acpp_sscp_kernel
// CHECK-NOT: land.lhs.true
// CHECK: select i1
// CHECK-NOT: land.lhs.true
// CHECK: ret void
