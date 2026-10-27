// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=1 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=4 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s

// Lanes of a sub-group pass values around through local memory, synchronized only by
// group_barrier(sub_group), in a loop. Work-groups of 20 work-items: at size 8 the last sub-group
// is a partial one of 4.

#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  const int S = queue.get_device().get_info<sycl::info::device::sub_group_sizes>().at(0);
  constexpr int wg = 20, n = 2 * wg, rounds = 3;
  int *out = sycl::malloc_shared<int>(n, queue);

  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<int> slots{wg, cgh};
    cgh.parallel_for(sycl::nd_range<1>{n, wg}, [=](sycl::nd_item<1> item) {
      auto sg = item.get_sub_group();
      const int lid = item.get_local_linear_id();
      const int base = lid - sg.get_local_linear_id(), size = sg.get_local_linear_range();
      slots[lid] = -1;
      sycl::group_barrier(item.get_group());

      int v = 100 * item.get_global_linear_id() + 1;
      for (int r = 0; r < rounds; ++r) {
        slots[lid] = v;
        sycl::group_barrier(sg); // all lanes have written
        v = slots[base + (sg.get_local_linear_id() + 1) % size];
        sycl::group_barrier(sg); // all lanes have read before the next round writes
      }
      out[item.get_global_linear_id()] = v;
    });
  });
  queue.wait();

  for (int i = 0; i < n; ++i) {
    const int l = i % wg, base = l / S * S, size = std::min(S, wg - base);
    const int expected = 100 * (i - l + base + (l - base + rounds) % size) + 1;
    if (out[i] != expected) {
      std::cout << "FAILED: S = " << S << ", work-item " << i << ": got " << out[i]
                << ", expected " << expected << '\n';
      return 1;
    }
  }
  std::cout << "PASSED\n";
}

// CHECK: PASSED
