
#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

#ifndef __SYCU__
#define __device__
#endif

int main()
{
  std::size_t num_threads = 128;

  cl::sycl::queue q;
  q.submit([&](cl::sycl::handler& cgh) {

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
  });

}
