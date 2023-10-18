// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=omp --acpp-use-accelerated-cpu -O
// RUN: %t | FileCheck %s

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

int main()
{
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i)
  {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};

    queue.submit([&](cl::sycl::handler &cgh) {
      using namespace cl::sycl::access;
      auto acc = buf.get_access<mode::read_write>(cgh);
      auto scratch = cl::sycl::accessor<int, 1, mode::read_write, target::local>{local_size, cgh};

      cgh.parallel_for<class dynamic_local_memory_reduction>(
        cl::sycl::nd_range<1>{global_size, local_size},
        [=](cl::sycl::nd_item<1> item) noexcept {
          const auto lid = item.get_local_id(0);
          const auto group_size = item.get_local_range(0);

          scratch[lid] = acc[item.get_global_id()];
          item.barrier();

          for(size_t i = lid; i < group_size + lid; i++)
          {
            scratch[lid] += i;
            // I am aware, that these barriers make no sense, I just couldn't come up with a useful algo using the lid..
            item.barrier();
          }

          if(lid == 0)
            acc[item.get_global_id()] = scratch[lid];
        });
    });
  }
  for(size_t i = 0; i < global_size / local_size; ++i)
  {
    // CHECK: 32640
    // CHECK: 32896
    // CHECK: 33152
    // CHECK: 33408
    std::cout << host_buf[i * local_size] << "\n";
  }
}
