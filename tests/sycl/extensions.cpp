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

BOOST_FIXTURE_TEST_SUITE(extension_tests, reset_device_fixture)

#ifdef HIPSYCL_EXT_AUTO_PLACEHOLDER_REQUIRE
BOOST_AUTO_TEST_CASE(auto_placeholder_require_extension) {
  namespace s = cl::sycl;

  s::queue q;
  s::buffer<int, 1> buff{1};
  s::accessor<int, 1, s::access::mode::read_write, 
    s::access::target::global_buffer, s::access::placeholder::true_t> acc{buff};

  // This will call handler::require(acc) for each
  // subsequently launched command group
  auto automatic_requirement = s::vendor::hipsycl::automatic_require(q, acc);
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&](s::handler &cgh) {
    cgh.single_task<class auto_require_kernel0>([=]() {
      acc[0] = 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 1);
  }

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel1>([=] (){
      acc[0] = 2;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 2);
  }

  automatic_requirement.release();
  BOOST_CHECK(!automatic_requirement.is_required());

  { 
    auto host_acc = buff.get_access<s::access::mode::read_write>(); 
    host_acc[0] = 3;
  }

  automatic_requirement.reacquire();
  BOOST_CHECK(automatic_requirement.is_required());

  q.submit([&] (s::handler& cgh) {
    cgh.single_task<class auto_require_kernel2>([=] (){
      acc[0] += 1;
    });
  });

  { 
    auto host_acc = buff.get_access<s::access::mode::read>(); 
    BOOST_CHECK(host_acc[0] == 4);
  }
}
#endif
#ifdef HIPSYCL_EXT_CUSTOM_PFWI_SYNCHRONIZATION
BOOST_AUTO_TEST_CASE(custom_pfwi_synchronization_extension) {
  namespace sync = cl::sycl::vendor::hipsycl::synchronization;

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
      auto scratch =
          cl::sycl::accessor<int, 1, cl::sycl::access::mode::read_write,
                             cl::sycl::access::target::local>{local_size,
                                                                    cgh};

      cgh.parallel_for_work_group<class pfwi_dispatch>(
        cl::sycl::range<1>{global_size / local_size},
        cl::sycl::range<1>{local_size},
        [=](cl::sycl::group<1> wg) {

          wg.parallel_for_work_item<sync::local_barrier>(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] = acc[item.get_global_id()];
          });

          // By default, a barrier is used
          wg.parallel_for_work_item(
            [&](cl::sycl::h_item<1> item) {
            scratch[item.get_local_id()[0]] *= 2;
          });

          // Testing the behavior of mem_fence() or 
          // that there is no synchronization is difficult,
          // so let's just test that things compile for now.
          wg.parallel_for_work_item<sync::none>(
            [&](cl::sycl::h_item<1> item) {
            acc[item.get_global_id()] = scratch[item.get_local_id()[0]];
          });

          wg.parallel_for_work_item<sync::local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });

          wg.parallel_for_work_item<sync::global_and_local_mem_fence>(
            [&](cl::sycl::h_item<1> item) {
          });
        });
    });
  }

  for(size_t i = 0; i < global_size; ++i) {
    BOOST_TEST(host_buf[i] == 2*i);
  }
}
#endif
#ifdef HIPSYCL_EXT_SCOPED_PARALLELISM
BOOST_AUTO_TEST_CASE(scoped_parallelism_reduction) {
  namespace s = cl::sycl;
  s::queue q;
  
  std::size_t input_size = 256;
  std::vector<int> input(input_size);
  for(int i = 0; i < input.size(); ++i)
    input[i] = i;
  
  s::buffer<int> buff{input.data(), s::range<1>{input_size}};
  
  constexpr size_t Group_size = 64;
  
  q.submit([&](s::handler& cgh){
    auto data_accessor = buff.get_access<s::access::mode::read_write>(cgh);
    cgh.parallel<class Kernel>(s::range<1>{input_size / Group_size}, s::range<1>{Group_size}, 
    [=](s::group<1> grp, s::physical_item<1> phys_idx){
      s::local_memory<int [Group_size]> scratch{grp};
      
      grp.distribute_for([&](s::sub_group sg, s::logical_item<1> idx){
          scratch[idx.get_local_id(0)] = data_accessor[idx.get_global_id(0)];
      });

      for(int i = Group_size / 2; i > 0; i /= 2){
        grp.distribute_for([&](s::sub_group sg, s::logical_item<1> idx){
          size_t lid = idx.get_local_id(0);
          if(lid < i)
            scratch[lid] += scratch[lid+i];
        });
      }
      
      grp.single_item([&](){
        data_accessor[grp.get_id(0)*Group_size] = scratch[0];
      });
    });
  });
  
  auto host_acc = buff.get_access<s::access::mode::read>();
  
  for(int grp = 0; grp < input_size/Group_size; ++grp){
    int host_result = 0;
    for(int i = grp * Group_size; i < (grp+1) * Group_size; ++i)
      host_result += i;
    
    BOOST_TEST(host_result == host_acc[grp * Group_size]);
  }
}


#endif

BOOST_AUTO_TEST_SUITE_END()