/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
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

#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(kernel_invocation_tests, reset_device_fixture)


BOOST_AUTO_TEST_CASE(basic_single_task) {
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(1)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.single_task<class basic_single_task>([=]() {
      acc[0] = 321;
    });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  BOOST_TEST(acc[0] == 321);
}

BOOST_AUTO_TEST_CASE(basic_parallel_for) {
  constexpr size_t num_threads = 128;
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(num_threads)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cgh.parallel_for<class basic_parallel_for>(cl::sycl::range<1>(num_threads),
      [=](cl::sycl::item<1> tid) {
        acc[tid] = tid[0];
      });
    });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads; ++i) {
    BOOST_REQUIRE(acc[i] == i);
  }
}

BOOST_AUTO_TEST_CASE(basic_parallel_for_with_offset) {
  constexpr size_t num_threads = 128;
  constexpr size_t offset = 64;
  cl::sycl::queue queue;
  std::vector<int> host_buf(num_threads + offset, 0);
  cl::sycl::buffer<int, 1> buf{host_buf.data(),
    cl::sycl::range<1>(num_threads + offset)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class basic_parallel_for_with_offset>(
      cl::sycl::range<1>(num_threads),
      cl::sycl::id<1>(offset),
      [=](cl::sycl::item<1> tid) {
        acc[tid] = tid[0];
      });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads + offset; ++i) {
    BOOST_REQUIRE(acc[i] == (i >= offset ? i : 0));
  }
}

BOOST_AUTO_TEST_CASE(basic_parallel_for_nd) {
  constexpr size_t num_threads = 128;
  constexpr size_t group_size = 16;
  cl::sycl::queue queue;
  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>(num_threads)};
  queue.submit([&](cl::sycl::handler& cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);
    cl::sycl::nd_range<1> kernel_range{cl::sycl::range<1>(num_threads),
      cl::sycl::range<1>(group_size)};
    cgh.parallel_for<class basic_parallel_for_nd>(kernel_range,
      [=](cl::sycl::nd_item<1> tid) {
        acc[tid.get_global_id()[0]] = tid.get_group(0);
      });
  });
  auto acc = buf.get_access<cl::sycl::access::mode::read>();
  for(int i = 0; i < num_threads; ++i) {
    BOOST_REQUIRE(acc[i] == i / group_size);
  }
}

BOOST_AUTO_TEST_CASE(hierarchical_dispatch) {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;
  std::vector<int> host_buf;
  for(size_t i = 0; i < global_size; ++i) {
    host_buf.push_back(static_cast<int>(i));
  }

  {
    cl::sycl::buffer<int, 1> buf{host_buf.data(), host_buf.size()};
    queue.submit([&](cl::sycl::handler& cgh) {
      auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);
      cgh.parallel_for_work_group<class hierarchical_dispatch_reduction>(
        cl::sycl::range<1>{global_size / local_size},
        cl::sycl::range<1>{local_size},
        [=](cl::sycl::group<1> wg) {
          int scratch[local_size];
          wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] = acc[item.get_global_id()];
          });
          for(size_t i = local_size/2; i > 0; i /= 2) {
            wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
              const size_t lid = item.get_local_id()[0];
              if(lid < i) scratch[lid] += scratch[lid + i];
            });
          }
          wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
            const size_t lid = item.get_local_id()[0];
            if(lid == 0) acc[item.get_global_id()] = scratch[0];
          });
        });
    });
  }

  for(size_t i = 0; i < global_size / local_size; ++i) {
    size_t expected = 0;
    for(size_t j = 0; j < local_size; ++j) expected += i * local_size + j;
    size_t computed = host_buf[i * local_size];
    BOOST_TEST(computed == expected);
  }
}

// hierarchical (especially private_memory) is not supported on nvc++
#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX 
BOOST_AUTO_TEST_CASE(hierarchical_private_memory) {
  constexpr size_t local_size = 256;
  constexpr size_t global_size = 1024;

  cl::sycl::queue queue;

  cl::sycl::buffer<int, 1> buf{cl::sycl::range<1>{global_size}};

  queue.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::discard_write>(cgh);

    cgh.parallel_for_work_group<class private_memory>(
      cl::sycl::range<1>{global_size / local_size},
      cl::sycl::range<1>{local_size}, [=](cl::sycl::group<1> wg) {

        cl::sycl::private_memory<int> my_int{wg};

        wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
          my_int(item) = item.get_global_id(0);
        });
        // Tests that private_memory is persistent across multiple
        // parallel_for_work_item() invocations
        wg.parallel_for_work_item([&](cl::sycl::h_item<1> item) {
          acc[item.get_global_id()] = my_int(item);
        });
      });
  });

  auto host_acc = buf.get_access<cl::sycl::access::mode::read>();
  for (int i = 0; i < global_size; ++i)
    BOOST_TEST(host_acc[i] == i);
}
#endif

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this
                            // line
