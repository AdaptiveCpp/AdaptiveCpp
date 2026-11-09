// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O
// RUN: env ACPP_VISIBILITY_MASK=omp %t | FileCheck %s

#include <cstddef>
#include <iostream>

#include <sycl/sycl.hpp>

int main() {
  sycl::queue queue{sycl::property::queue::in_order{}};

  using value_type = double;
  constexpr std::size_t local_size = 32;
  constexpr std::size_t num_groups = 2;
  constexpr std::size_t num_values = 16;
  constexpr std::size_t global_size = num_groups * local_size;

  auto *output =
      sycl::malloc_device<value_type>(num_groups * num_values, queue);

  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
          value_type input[num_values];
          for (std::size_t k = 0; k < num_values; ++k)
            input[k] = item.get_local_linear_id() + k;

          value_type reduced[num_values];
          for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j)
              reduced[i * 4 + j] = sycl::reduce_over_group(
                  item.get_group(), input[i * 4 + j], sycl::plus<value_type>{});

          if (item.get_group().leader()) {
            const auto group = item.get_group().get_group_linear_id();
            for (std::size_t k = 0; k < num_values; ++k)
              output[group * num_values + k] = reduced[k];
          }
        });
  });

  value_type result[num_groups * num_values];
  queue.memcpy(result, output, sizeof(result)).wait_and_throw();
  sycl::free(output, queue);

  for (std::size_t group = 0; group < num_groups; ++group) {
    for (std::size_t k = 0; k < num_values; ++k) {
      const value_type expected = 496 + local_size * k;
      if (result[group * num_values + k] != expected) {
        std::cout << "FAILED: group " << group << ", element " << k << ", got "
                  << result[group * num_values + k] << ", expected " << expected
                  << '\n';
        return 1;
      }
    }
  }

  std::cout << "PASSED\n";
}

// CHECK: PASSED
