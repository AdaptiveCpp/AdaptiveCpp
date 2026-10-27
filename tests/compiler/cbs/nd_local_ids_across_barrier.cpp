// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

// Values computed from the local ids of several dimensions before a barrier and used after it.
// Only the innermost dimension's id steps by one between neighbouring work-items, so e.g. y - x
// and y <= x differ between work-items and must be kept per work-item across the barrier.

#include <cstddef>
#include <iostream>

#include <sycl/sycl.hpp>

template <int Dim> int value(const sycl::id<Dim> &lid) {
  const int x = lid[Dim - 1], y = lid[0], m = lid[Dim / 2];
  return 1000 * (y <= x) + 100 * (m <= x) + (y - x);
}

template <int Dim> bool test(sycl::queue &queue, sycl::range<Dim> size) {
  const std::size_t n = size.size();
  int *pre = sycl::malloc_shared<int>(n, queue);
  int *out = sycl::malloc_shared<int>(n, queue);

  queue.parallel_for(sycl::nd_range<Dim>{size, size}, [=](sycl::nd_item<Dim> item) {
    const std::size_t i = item.get_local_linear_id();
    const int v = value(item.get_local_id());
    pre[i] = v;
    sycl::group_barrier(item.get_group());
    out[i] = v - pre[n - 1 - i];
  });
  queue.wait();

  bool ok = true;
  auto lid = [&](std::size_t i) {
    sycl::id<Dim> id;
    for (int d = Dim - 1; d >= 0; --d) {
      id[d] = i % size[d];
      i /= size[d];
    }
    return id;
  };
  for (std::size_t i = 0; i < n && ok; ++i) {
    const int expected = value(lid(i)) - value(lid(n - 1 - i));
    if (out[i] != expected) {
      std::cout << "FAILED: " << Dim << "D, work-item " << i << ", got " << out[i]
                << ", expected " << expected << '\n';
      ok = false;
    }
  }
  sycl::free(pre, queue);
  sycl::free(out, queue);
  return ok;
}

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};
  if (!test(queue, sycl::range<2>{8, 8}) || !test(queue, sycl::range<3>{4, 4, 4}))
    return 1;
  std::cout << "PASSED\n";
}

// CHECK: PASSED
