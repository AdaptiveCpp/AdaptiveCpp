/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cassert>
#include <iostream>

#include <CL/sycl.hpp>

#ifndef __HIPSYCL__
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
        printf("Hello from sycl item %lu\n", tid.get_linear_id());
    });

    // Test with id class
    cgh.parallel_for<class hello_world_id>(cl::sycl::range<1>{num_threads},
                                           [=] __device__ (cl::sycl::id<1> tid) {
      if(tid[0] == 0)
        printf("Hello from sycl id %lu\n", tid[0]);
    });

    // Test with nd_item
    cgh.parallel_for<class hello_world_ndrange>(cl::sycl::nd_range<>(cl::sycl::range<1>(num_threads),
                                                                     cl::sycl::range<1>(group_size)),
                                                [=] __device__ (cl::sycl::nd_item<1> tid){
      size_t lid = tid.get_local(0);
      size_t group_id = tid.get_group(0);

      if(lid == 0)
        printf("Hello world from group %lu\n", group_id);
    });

    // Test with offset
    cgh.parallel_for<class hello_world_offset>(cl::sycl::range<2>{16,16},
                                               cl::sycl::id<2>{2,3},
                                               [=] __device__ (cl::sycl::item<2> tid){
      printf("Hello world with 2d offset from thread %lu,%lu, linear id %lu\n",
             tid.get_id(0), tid.get_id(1), tid.get_linear_id());
    });
  });

}
