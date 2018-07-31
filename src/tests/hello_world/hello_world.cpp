
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
    cgh.parallel_for<class hello_world>(cl::sycl::range<1>{num_threads},
                                        [=] __device__ (cl::sycl::item<1> tid){
      if(tid.get_linear_id() == 0)
        printf("Hello from thread %d\n", tid.get_linear_id());
    });
  });

}
