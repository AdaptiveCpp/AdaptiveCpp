// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <cstdint>
#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};

  constexpr std::size_t local_size = 256;
  constexpr std::size_t num_groups = 4;
  constexpr std::uint32_t num_items = 16;

  auto *input = sycl::malloc_shared<std::uint32_t>(num_items, queue);
  auto *output = sycl::malloc_shared<std::uint32_t>(num_items, queue);
  for (std::uint32_t i = 0; i < num_items; ++i) {
    input[i] = 100 + i;
    output[i] = 0;
  }

  queue.parallel_for(
      sycl::nd_range<1>{num_groups * local_size, local_size},
      [=](sycl::nd_item<1> item) {
        for (std::uint32_t i = item.get_group(0); i < num_items;
             i += num_groups) {
          std::uint32_t n = input[i];
          sycl::group_barrier(item.get_group());
          if (item.get_group().leader())
            output[i] = n;
        }
      });
  queue.wait_and_throw();

  int failures = 0;
  for (std::uint32_t i = 0; i < num_items; ++i) {
    if (output[i] != input[i]) {
      std::cout << "FAILED: item " << i << ", got " << output[i]
                << ", expected " << input[i] << '\n';
      ++failures;
    }
  }
  sycl::free(input, queue);
  sycl::free(output, queue);

  if (failures == 0)
    std::cout << "PASSED\n";
  return failures != 0;
}

// CHECK: PASSED
