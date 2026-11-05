// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=2 ACPP_APPDB_DIR=%t.db %t | FileCheck %s
// RUN: rm -rf %t.db
// RUN: env ACPP_VISIBILITY_MASK=omp ACPP_OPENMP_SUBGROUP_SIZE=4 ACPP_APPDB_DIR=%t.db %t | FileCheck %s

// Work-group reduce with more sub-groups than its 32 partial results (S = 2 or 4 in work-groups
// of 144 and 256): the sub-groups past the 32nd are added into the partial results, once per
// sub-group, not once per lane (on the host the lanes run one after another).

#include <cstdio>

#include <sycl/sycl.hpp>

// CHECK: L=25: ok
// CHECK: L=144: ok
// CHECK: L=256: ok
int main() {
  sycl::queue q;
  for (int L : {25, 144, 256}) {
    const int G = 3, N = G * L;
    int *In = sycl::malloc_shared<int>(N, q), *Out = sycl::malloc_shared<int>(N, q);
    for (int i = 0; i < N; ++i)
      In[i] = i % 7 + 1;
    q.parallel_for(sycl::nd_range<1>{size_t(N), size_t(L)}, [=](sycl::nd_item<1> it) {
      const size_t i = it.get_global_id(0);
      Out[i] = sycl::reduce_over_group(it.get_group(), In[i], sycl::plus<int>());
    }).wait();
    int Bad = 0;
    for (int g = 0; g < G; ++g) {
      int E = 0;
      for (int j = 0; j < L; ++j)
        E += In[g * L + j];
      for (int j = 0; j < L; ++j)
        Bad += Out[g * L + j] != E;
    }
    std::printf("L=%d: %s\n", L, Bad ? "FAIL" : "ok");
    sycl::free(In, q);
    sycl::free(Out, q);
  }
}
