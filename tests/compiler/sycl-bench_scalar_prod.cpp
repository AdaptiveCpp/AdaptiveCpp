// The reduction in the original code has a WAR dependency between work groups -> flaky territory..
// ALLOW_RETRIES: 4
// RUN: %syclcc %s -o %t --hipsycl-targets=omp -DHIPSYCL_NO_FIBERS
// RUN: %t | FileCheck %s
// RUN: %syclcc %s -o %t --hipsycl-targets=omp -DHIPSYCL_NO_FIBERS -O
// RUN: %t | FileCheck %s

#include <array>
#include <iostream>

#include <CL/sycl.hpp>

namespace s = cl::sycl;
int main()
{
  constexpr size_t local_size = 64;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;

  std::vector<int> input1(global_size);
  std::vector<int> input2(global_size);
  std::vector<int> output(global_size);
  for(size_t i = 0; i < global_size; ++i)
  {
    input1[i] = 1;
    input2[i] = 2;
  }

  {
    cl::sycl::buffer<int, 1> input1_buf{input1.data(), input1.size()};
    cl::sycl::buffer<int, 1> input2_buf{input2.data(), input2.size()};
    cl::sycl::buffer<int, 1> output_buf{output.data(), output.size()};

    auto array_size = global_size;
    auto wgroup_size = local_size;
    queue.submit([&](cl::sycl::handler &cgh) {
      auto sum = output_buf.template get_access<s::access::mode::discard_write>(cgh);
      auto i1 = input1_buf.template get_access<s::access::mode::read>(cgh);
      auto i2 = input2_buf.template get_access<s::access::mode::read>(cgh);
      cgh.parallel_for(s::range<1>{static_cast<size_t>(array_size)}, [=](cl::sycl::id<1> id) {
        sum[id] = i1[id] * i2[id];
      });
    });

    // Not yet tested with more than 2
    auto elements_per_thread = 2;

    while(array_size != 1)
    {
      using T = int;
      auto n_wgroups = (array_size + wgroup_size * elements_per_thread - 1) / (wgroup_size * elements_per_thread); // two threads per work item
      queue.submit([&](cl::sycl::handler &cgh) {
        auto global_mem =
          output_buf.template get_access<s::access::mode::read_write>(cgh);

        // local memory for reduction
        auto local_mem =
          s::accessor<T, 1, s::access::mode::read_write,
            s::access::target::local>{s::range<1>(wgroup_size), cgh};
        cl::sycl::nd_range<1> ndrange(n_wgroups * wgroup_size, wgroup_size);

        cgh.parallel_for<class ScalarProdReduction>(ndrange,
          [=](cl::sycl::nd_item<1> item) {
            size_t gid = item.get_global_linear_id();
            size_t lid = item.get_local_linear_id();

            // initialize local memory to 0
            local_mem[lid] = 0;

            if((elements_per_thread * gid) < array_size)
            {
              local_mem[lid] = global_mem[elements_per_thread * gid] + global_mem[elements_per_thread * gid + 1];
            }

            item.barrier(s::access::fence_space::local_space);

            for(size_t stride = 1; stride < wgroup_size;
                stride *= elements_per_thread)
            {
              auto local_mem_index = elements_per_thread * stride * lid;
              if(local_mem_index < wgroup_size)
              {
                local_mem[local_mem_index] =
                  local_mem[local_mem_index] + local_mem[local_mem_index + stride];
              }

              item.barrier(s::access::fence_space::local_space);
            }

            // Only one work-item per work group writes to global memory
            if(lid == 0)
            {
              global_mem[item.get_group_linear_id()] = local_mem[0];
            }
          });
      });
      array_size = n_wgroups;
    }
  }
  // CHECK: 2048
  std::cout << output[0] << std::endl;
}
