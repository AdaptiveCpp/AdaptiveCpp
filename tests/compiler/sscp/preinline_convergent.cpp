// The simplification before inlining must not duplicate a barrier. The block after the second
// barrier branches on the same condition as the block before it; threading the barrier into both
// predecessors would leave the items below and above 32 at different barriers.
//
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -f %t.ll
// RUN: env ACPP_S2_DUMP_IR_JIT_OPTIMIZATIONS=%t.ll %t | FileCheck %s --check-prefix=OUT
// RUN: FileCheck %s --input-file %t.ll

#include "common.hpp"
#include <iostream>
#include <vector>

int main() {
  sycl::queue q = get_queue();
  constexpr int local_size = 128;
  std::vector<int> host_out(local_size, -1);
  int *out = sycl::malloc_device<int>(local_size, q);
  q.memcpy(out, host_out.data(), local_size * sizeof(int)).wait();

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int> scratch{local_size, cgh};
    cgh.parallel_for(sycl::nd_range<1>{local_size, local_size}, [=](sycl::nd_item<1> item) {
      const int lid = item.get_local_id(0);
      scratch[lid] = lid;
      item.barrier(sycl::access::fence_space::local_space);
      if (lid < 32)
        scratch[lid] += scratch[lid + 32];
      item.barrier(sycl::access::fence_space::local_space);
      if (lid < 32)
        out[lid] = scratch[lid];
    });
  }).wait();
  q.memcpy(host_out.data(), out, local_size * sizeof(int)).wait();

  int sum = 0;
  for (int i = 0; i < 32; ++i)
    sum += host_out[i];
  // sum over i < 32 of i + (i + 32)
  // OUT: 2016
  std::cout << sum << std::endl;
  sycl::free(out, q);
  return 0;
}

// CHECK-LABEL: stage: jit_optimizations
// CHECK-LABEL: define {{.*}}__acpp_sscp_kernel
// CHECK-COUNT-2: call void @__acpp_sscp_work_group_barrier
// CHECK-NOT: call void @__acpp_sscp_work_group_barrier
// CHECK: ret void
