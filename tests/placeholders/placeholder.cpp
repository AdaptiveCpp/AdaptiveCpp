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
#include <cassert>

#include <CL/sycl.hpp>


constexpr std::size_t num_elements  = 4096*1024;

int main()
{
  cl::sycl::queue q;

  cl::sycl::buffer<int,1> buff_a{num_elements};

  {
    auto host_access = buff_a.get_access<cl::sycl::access::mode::discard_write>();

    for(std::size_t i = 0; i < num_elements; ++i)
      host_access[i] = static_cast<int>(i);
  }

  cl::sycl::accessor<int,1,
                     cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::global_buffer,
                     cl::sycl::access::placeholder::true_t>
      placeholder_accessor{buff_a};

  q.submit([&](cl::sycl::handler& cgh) {
    cgh.require(placeholder_accessor);
    // Test copying accessors
    auto placeholder_copy = placeholder_accessor;
    cgh.require(placeholder_copy);

    cgh.parallel_for<class placeholder_test>(cl::sycl::range<1>{num_elements},
          [=] (cl::sycl::id<1> tid) {

      placeholder_accessor[tid] *= 2;
    });
  });

  {
    auto host_access = buff_a.get_access<cl::sycl::access::mode::read>();

    for(std::size_t i = 0; i < num_elements; ++i)
      assert(host_access[i] == 2*i);
  }

  std::cout << "Passed." << std::endl;
}


