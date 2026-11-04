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

// A contiguous value (a uniform loop variable plus the local id) computed before a barrier in a
// loop and used after it. CBS recomputes such values after the barrier instead of keeping them
// per work-item; the region after the barrier also reaches the value's definition again through
// the loop.

#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  constexpr int local_size = 16, rounds = 3;
  int *out = sycl::malloc_shared<int>(local_size * rounds, queue);

  queue.parallel_for(sycl::nd_range<1>{local_size, local_size}, [=](sycl::nd_item<1> item) {
    const int j = item.get_local_linear_id();
    for (int b = 0; b < rounds * local_size; b += local_size) {
      const int c = b + j;
      sycl::group_barrier(item.get_group());
      out[c] = c;
    }
  });
  queue.wait();

  for (int i = 0; i < local_size * rounds; ++i)
    if (out[i] != i) {
      std::cout << "FAILED: element " << i << ": got " << out[i] << '\n';
      return 1;
    }
  std::cout << "PASSED\n";
}

// CHECK: PASSED
