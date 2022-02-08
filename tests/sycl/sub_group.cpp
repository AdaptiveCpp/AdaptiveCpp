/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2020 Aksel Alpay and contributors
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


#include "hipSYCL/sycl/info/device.hpp"
#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(sub_group_tests, reset_device_fixture)



BOOST_AUTO_TEST_CASE(sub_group) {
  namespace s = cl::sycl;
  s::queue q;
  s::range<1> size1d{1024};
  s::range<2> size2d{32, 32};
  s::range<3> size3d{16, 16, 16};

  s::range<1> local_size1d{128};
  s::range<2> local_size2d{16, 16};
  s::range<3> local_size3d{4, 8, 8};

  s::buffer<uint32_t, 1> buff1d{size1d};
  s::buffer<uint32_t, 2> buff2d{size2d};
  s::buffer<uint32_t, 3> buff3d{size3d};

  q.submit([&](s::handler &cgh) {
    auto acc = buff1d.get_access<s::access::mode::discard_write>(cgh);
    cgh.parallel_for<class sub_group1d>(
        s::nd_range<1>{size1d, local_size1d}, [=](s::nd_item<1> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();
    });
  });

  q.submit([&](s::handler &cgh) {
    auto acc = buff2d.get_access<s::access::mode::discard_write>(cgh);
    cgh.parallel_for<class sub_group2d>(
        s::nd_range<2>{size2d, local_size2d}, [=](s::nd_item<2> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();
    });
  });

  q.submit([&](s::handler &cgh) {
    auto acc = buff3d.get_access<s::access::mode::discard_write>(cgh);
    cgh.parallel_for<class sub_group3d>(
        s::nd_range<3>{size3d, local_size3d}, [=](s::nd_item<3> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();
    });
  });

  q.wait_and_throw();
  auto host_acc1 = buff1d.get_access<s::access::mode::read>();
  auto host_acc2 = buff2d.get_access<s::access::mode::read>();
  auto host_acc3 = buff3d.get_access<s::access::mode::read>();

  const s::device dev = q.get_device();
  const std::vector<size_t> supported_subgroup_sizes =
      dev.get_info<cl::sycl::info::device::sub_group_sizes>();
  BOOST_CHECK(supported_subgroup_sizes.size() >= 1);
  const unsigned int max_num_subgroups =
      dev.get_info<cl::sycl::info::device::max_num_sub_groups>();
  BOOST_CHECK(max_num_subgroups >= 1U);

  uint32_t subgroup_size = supported_subgroup_sizes[0];

  for (size_t i = 0; i < size1d[0]; ++i) {
    size_t lid = i % local_size1d[0];
    BOOST_TEST_INFO("i: " << i);
    BOOST_CHECK_EQUAL(host_acc1[i], lid % subgroup_size);
  }
  for (size_t i = 0; i < size2d[0]; ++i) {
    for (size_t j = 0; j < size2d[1]; ++j) {
      auto id = s::id<2>{i, j};
      auto lid = id % local_size2d;
      BOOST_CHECK_EQUAL(host_acc2[id], (lid[1] + lid[0]*local_size2d[1]) % subgroup_size);
    }
  }
  for (size_t i = 0; i < size3d[0]; ++i) {
    for (size_t j = 0; j < size3d[1]; ++j) {
      for (size_t k = 0; k < size3d[2]; ++k) {
        auto id = s::id<3>{i, j, k};
        auto lid = id % local_size3d;
        BOOST_CHECK_EQUAL(host_acc3[id],
                    (lid[2] + lid[1] * local_size3d[2] +
                     lid[0] * local_size3d[1] * local_size3d[2]) %
                        subgroup_size);
      }
    }
  }
}



BOOST_AUTO_TEST_SUITE_END()
