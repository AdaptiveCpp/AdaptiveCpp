// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_ADAPTIVITY_LEVEL=0 %t | FileCheck %s

// Barriers in an inner loop, skipped by `continue` depending on a mask loaded in the outer loop
// (the same for all work-items). In outer iteration 1 the mask is 0, so the inner loop passes no
// barrier: the region from the last barrier of iteration 0 to the first of iteration 2 runs both
// loops and must carry the mask, the counters and acc through their loop headers.

#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  constexpr int local_size = 8, outer = 3, inner = 4;
  unsigned *mask = sycl::malloc_shared<unsigned>(outer, queue);
  int *out = sycl::malloc_shared<int>(local_size, queue);
  for (int j = 0; j < outer; ++j)
    mask[j] = j == 1 ? 0u : 0xfu;

  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int> slots{local_size, cgh};
    cgh.parallel_for(sycl::nd_range<1>{local_size, local_size}, [=](sycl::nd_item<1> item) {
      const int lid = item.get_local_linear_id();
      int acc = 0;
#pragma clang loop unroll(disable)
      for (int j = 0; j < outer; ++j) {
        const unsigned m = mask[j];
#pragma clang loop unroll(disable)
        for (int k = 0; k < inner; ++k) {
          if (!(m & (1u << k)))
            continue;
          slots[lid] = lid + k;
          sycl::group_barrier(item.get_group());
          acc += slots[lid ^ 1] + 1;
          sycl::group_barrier(item.get_group());
        }
      }
      out[lid] = acc;
    });
  });
  queue.wait();

  for (int i = 0; i < local_size; ++i) {
    int expected = 0;
    for (int j = 0; j < outer; ++j)
      for (int k = 0; k < inner; ++k)
        if (mask[j] & (1u << k))
          expected += (i ^ 1) + k + 1;
    if (out[i] != expected) {
      std::cout << "FAILED: work-item " << i << ": got " << out[i] << ", expected " << expected
                << '\n';
      return 1;
    }
  }
  std::cout << "PASSED\n";
}

// CHECK: PASSED
