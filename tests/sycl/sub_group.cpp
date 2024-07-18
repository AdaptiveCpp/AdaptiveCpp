/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause


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

  s::buffer<uint32_t, 1> used_sgrp_sizes{3};

  q.submit([&](s::handler &cgh) {
    auto acc = buff1d.get_access<s::access::mode::discard_write>(cgh);
    auto sgrp_size = used_sgrp_sizes.get_access<s::access::mode::write>(cgh);

    cgh.parallel_for<class sub_group1d>(
        s::nd_range<1>{size1d, local_size1d}, [=](s::nd_item<1> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();

      if(idx.get_global_linear_id() == 0)
        sgrp_size[0] = sgrp.get_local_linear_range();
    });
  });

  q.submit([&](s::handler &cgh) {
    auto acc = buff2d.get_access<s::access::mode::discard_write>(cgh);
    auto sgrp_size = used_sgrp_sizes.get_access<s::access::mode::write>(cgh);

    cgh.parallel_for<class sub_group2d>(
        s::nd_range<2>{size2d, local_size2d}, [=](s::nd_item<2> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();

      if(idx.get_global_linear_id() == 0)
        sgrp_size[1] = sgrp.get_local_linear_range();
    });
  });

  q.submit([&](s::handler &cgh) {
    auto acc = buff3d.get_access<s::access::mode::discard_write>(cgh);
    auto sgrp_size = used_sgrp_sizes.get_access<s::access::mode::write>(cgh);

    cgh.parallel_for<class sub_group3d>(
        s::nd_range<3>{size3d, local_size3d}, [=](s::nd_item<3> idx) {
      s::sub_group sgrp = idx.get_sub_group();
      acc[idx.get_global_id()] = sgrp.get_local_linear_id();

      if(idx.get_global_linear_id() == 0)
        sgrp_size[2] = sgrp.get_local_linear_range();
    });
  });

  q.wait_and_throw();
  auto host_acc1 = buff1d.get_access<s::access::mode::read>();
  auto host_acc2 = buff2d.get_access<s::access::mode::read>();
  auto host_acc3 = buff3d.get_access<s::access::mode::read>();
  auto host_sgrp_sizes = used_sgrp_sizes.get_access<s::access::mode::read>();

  const s::device dev = q.get_device();
  const std::vector<size_t> supported_subgroup_sizes =
      dev.get_info<cl::sycl::info::device::sub_group_sizes>();
  BOOST_CHECK(supported_subgroup_sizes.size() >= 1);
  const unsigned int max_num_subgroups =
      dev.get_info<cl::sycl::info::device::max_num_sub_groups>();
  BOOST_CHECK(max_num_subgroups >= 1U);

  // check that subgroup size obtained from kernel is one of the sizes
  // listed as supported
  for(int i = 0; i < used_sgrp_sizes.size(); ++i) {
    auto size = host_sgrp_sizes[i];
    BOOST_CHECK(std::find(supported_subgroup_sizes.begin(),
                          supported_subgroup_sizes.end(),
                          size) != supported_subgroup_sizes.end());
  }

  uint32_t subgroup_size = supported_subgroup_sizes[0];

  for (size_t i = 0; i < size1d[0]; ++i) {
    size_t lid = i % local_size1d[0];
    BOOST_TEST_INFO("i: " << i);
    BOOST_CHECK_EQUAL(host_acc1[i], lid % host_sgrp_sizes[0]);
  }
  for (size_t i = 0; i < size2d[0]; ++i) {
    for (size_t j = 0; j < size2d[1]; ++j) {
      auto id = s::id<2>{i, j};
      auto lid = id % local_size2d;
      BOOST_CHECK_EQUAL(host_acc2[id], (lid[1] + lid[0]*local_size2d[1]) % host_sgrp_sizes[1]);
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
                        host_sgrp_sizes[2]);
      }
    }
  }
}



BOOST_AUTO_TEST_SUITE_END()
