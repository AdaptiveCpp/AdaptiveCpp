// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <cassert>
#include <sycl/sycl.hpp>
#include "common.hpp"


#include <sycl/sycl.hpp>

int main() {
  sycl::queue q = get_queue();

  int init = 32;
  auto const x = sycl::malloc_device<int>(1, q);
  q.memcpy(x, &init, sizeof(int)).wait();

  q.submit(
      [&](sycl::handler &cgh) { cgh.single_task([=]() { assert(x[0] == 32); x[0] += 1; }); });

  q.wait();

  int host;
  q.memcpy(&host, x, sizeof(int)).wait();

  // CHECK: 33
  std::cout << host << std::endl;

  sycl::free(x, q);
  return 0;
}
