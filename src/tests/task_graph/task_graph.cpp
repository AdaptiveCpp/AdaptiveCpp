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

#include <iostream>

#include <CL/sycl.hpp>

#ifndef __HIPSYCL__
#define __device__
#endif

constexpr std::size_t num_elements = 4096*1024;

int main()
{
  cl::sycl::queue q1;
  cl::sycl::queue q2;
  cl::sycl::queue q3;

  cl::sycl::buffer<int, 1> buff_a{num_elements};
  cl::sycl::buffer<int, 1> buff_b{num_elements};
  cl::sycl::buffer<int, 1> buff_c{num_elements};

  q1.submit([&](cl::sycl::handler& cgh){
    auto access_a =
        buff_a.get_access<cl::sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for<class init_a>(cl::sycl::range<1>{num_elements},
                                   [=] __device__ (cl::sycl::id<1> tid)
    {
      access_a[tid] = static_cast<int>(tid.get(0));
    });
  });

  q2.submit([&](cl::sycl::handler& cgh){
    auto access_b =
        buff_b.get_access<cl::sycl::access::mode::discard_write>(cgh);
    auto access_a =
        buff_a.get_access<cl::sycl::access::mode::read>(cgh);

    cgh.parallel_for<class init_b>(cl::sycl::range<1>{num_elements},
                                   [=] __device__ (cl::sycl::id<1> tid)
    {
      access_b[tid] = access_a[tid];
    });
  });

  q3.submit([&](cl::sycl::handler& cgh){
    auto access_a = buff_a.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_b = buff_b.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_c = buff_c.get_access<cl::sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<class add_a_b>(cl::sycl::range<1>{num_elements},
                                    [=] __device__ (cl::sycl::id<1> tid){
      access_c[tid] = access_a[tid] + access_b[tid];
    });
  });

  auto result = buff_c.get_access<cl::sycl::access::mode::read>();

  for(size_t i = num_elements - 10; i < num_elements; ++i)
    std::cout << result[i] << std::endl;

}

