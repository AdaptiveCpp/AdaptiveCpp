
#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

#ifndef __SYCU__
#define __device__
#endif

int main()
{
  std::size_t num_threads = 128;
  std::size_t group_size = 16;

  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler& cgh) {

    cgh.single_task<class hello_world_single_task>([=] __device__ (){
      printf("Hello world from single task!\n");
    });

    // First, test with item class
    cgh.parallel_for<class hello_world_item>(cl::sycl::range<1>{num_threads},
                                        [=] __device__ (cl::sycl::item<1> tid){
      if(tid.get_linear_id() == 0)
        printf("Hello from sycl item %u\n", tid.get_linear_id());
    });

    // Test with id class
    cgh.parallel_for<class hello_world_id>(cl::sycl::range<1>{num_threads},
                                           [=] __device__ (cl::sycl::id<1> tid) {
      if(tid[0] == 0)
        printf("Hello from sycl id %u\n", tid[0]);
    });

    cgh.parallel_for<class hello_world_ndrange>(cl::sycl::nd_range<>(cl::sycl::range<1>(num_threads),
                                                                     cl::sycl::range<1>(group_size)),
                                                [=](cl::sycl::nd_item<1> tid){

    });
  });

}
