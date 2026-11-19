// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=1 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=4 %t | FileCheck %s
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_APPDB_DIR=%t.db ACPP_OPENMP_SUBGROUP_SIZE=8 %t | FileCheck %s

// Sub-group ids and sizes on the OpenMP device with ACPP_OPENMP_SUBGROUP_SIZE: work-groups of 20
// work-items, so at size 8 the last sub-group is a partial one of 4. The runs share one JIT cache,
// so they also check that kernels for different sub-group sizes are kept apart.

#include <cstdlib>
#include <iostream>

#include <sycl/sycl.hpp>

template <int Dim>
bool test(sycl::queue &queue, int S, sycl::range<Dim> global, sycl::range<Dim> local) {
  const int n = global.size(), wg = local.size();
  int *r = sycl::malloc_shared<int>(n * 5, queue);
  queue.parallel_for(sycl::nd_range<Dim>{global, local}, [=](sycl::nd_item<Dim> item) {
    auto sg = item.get_sub_group();
    int *o = r + item.get_global_linear_id() * 5;
    o[0] = sg.get_local_linear_id();
    o[1] = sg.get_group_linear_id();
    o[2] = sg.get_local_linear_range();
    o[3] = sg.get_max_local_range()[0];
    o[4] = sg.get_group_linear_range();
  });
  queue.wait();

  bool ok = true;
  for (int i = 0; i < n && ok; ++i) {
    const int l = i % wg, num = (wg + S - 1) / S, id = l / S;
    const int expected[5] = {l % S, id, id == num - 1 ? wg - (num - 1) * S : S, S, num};
    for (int k = 0; k < 5; ++k)
      if (r[i * 5 + k] != expected[k]) {
        std::cout << "FAILED: " << Dim << "D, S = " << S << ", work-item " << i << ", value " << k
                  << ": got " << r[i * 5 + k] << ", expected " << expected[k] << '\n';
        ok = false;
      }
  }
  sycl::free(r, queue);
  return ok;
}

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  const int S = queue.get_device().get_info<sycl::info::device::sub_group_sizes>().at(0);
  if (S != std::atoi(std::getenv("ACPP_OPENMP_SUBGROUP_SIZE"))) {
    std::cout << "FAILED: the device reports sub-group size " << S << '\n';
    return 1;
  }
  if (!test(queue, S, sycl::range<1>{40}, sycl::range<1>{20}) ||
      !test(queue, S, sycl::range<2>{4, 10}, sycl::range<2>{2, 10}))
    return 1;
  std::cout << "PASSED\n";
}

// CHECK: PASSED
