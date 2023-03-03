// RUN: %syclcc %s -o %t --opensycl-targets=omp --opensycl-use-accelerated-cpu
// RUN: %t | FileCheck %s
// RUN: %syclcc %s -o %t --opensycl-targets=omp --opensycl-use-accelerated-cpu -O
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

          scratch[lid] = acc[item.get_global_id()];
          auto tmp = scratch[lid];
          for(size_t i = local_size / 2; i > 0; i /= 2)
          {
            item.barrier();
            if(lid < i)
              tmp += scratch[lid + i];
            scratch[lid] = tmp;
          }

          if(lid <= 1)
            acc[item.get_global_id()] = tmp;
        });
    });
  }
  for(size_t i = 0; i < global_size / local_size; ++i)
  {
    // CHECK: 16384
    // CHECK: 32640
    // CHECK: 49152
    // CHECK: 98176
    // CHECK: 81920
    // CHECK: 163712
    // CHECK: 114688
    // CHECK: 229248
    std::cout << host_buf[i * local_size + 1] << "\n";
    std::cout << host_buf[i * local_size] << "\n";
  }
}
