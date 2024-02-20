// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s

#include <iostream>

#include <sycl/sycl.hpp>

int main()
{
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i)
  {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](sycl::handler &cgh) {
      using namespace sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);

      cgh.parallel_for<class dynamic_local_memory_reduction>(
        sycl::nd_range<1>{global_size, local_size},
        [=](sycl::nd_item<1> item) noexcept {
          acc[item.get_global_id()] += 2;
        });
    });
  }
  for(size_t i = 0; i < global_size / local_size; ++i)
  {
    // CHECK: 2
    // CHECK: 258
    // CHECK: 514
    // CHECK: 770
    std::cout << host_buf[i * local_size] << "\n";
  }
}
