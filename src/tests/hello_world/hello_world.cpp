
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
                                        [=] __device__ (cl::sycl::id<1> tid){
      printf("Hello from thread %d", tid[0]);
    });
  });
}
