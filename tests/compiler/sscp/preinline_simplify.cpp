// Functions are simplified before being inlined into the kernel, so the
// `a && b && c` below must reach the JIT pipeline already folded into
// branchless code.
//
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -f %t.ll
// RUN: env ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS=%t.ll %t | FileCheck %s --check-prefix=OUT
// RUN: FileCheck %s --input-file %t.ll

#include "common.hpp"
#include <iostream>

struct bounds {
  int lo;
};

// A real function, so that its reference parameter carries `dereferenceable`.
// Every conjunct reads the same field, so each is a single compare once the
// redundant loads are gone.
static int classify(const bounds &b, int x, int y, int z) {
  if (x > b.lo && y > b.lo && z > b.lo)
    return 1;
  return 0;
}

int main() {
  sycl::queue q = get_queue();

  bounds *b = sycl::malloc_device<bounds>(1, q);
  int *v = sycl::malloc_device<int>(3, q);
  int *out = sycl::malloc_device<int>(1, q);
  const bounds host_b{2};
  const int host_v[3] = {3, 4, 5};
  q.memcpy(b, &host_b, sizeof(bounds)).wait();
  q.memcpy(v, host_v, sizeof(host_v)).wait();

  q.single_task([=]() { *out = classify(*b, v[0], v[1], v[2]); }).wait();

  int result = 0;
  q.memcpy(&result, out, sizeof(int)).wait();

  // OUT: 1
  std::cout << result << std::endl;

  sycl::free(b, q);
  sycl::free(v, q);
  sycl::free(out, q);
  return 0;
}

// The dump taken right after the always-inliner: the conjuncts must have been
// folded while `b` was still a `dereferenceable` argument, so the kernel is
// free of conditional branches.
//
// CHECK-LABEL: stage: jit_optimizations
// CHECK-LABEL: define {{.*}}__acpp_sscp_kernel
// CHECK-NOT: br i1
// CHECK: ret void
