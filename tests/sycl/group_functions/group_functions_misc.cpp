/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018-2020 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

/*
 * ND-range
 */

BOOST_AUTO_TEST_CASE(nd_range_group_barrier) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<int> out_data(global_size);

    {
      sycl::buffer<int, 1> out_buf{out_data.data(), out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor out_acc{out_buf, cgh, sycl::write_only};
        auto scratch = sycl::accessor<int, 1, mode::read_write, target::local>{1, cgh};

        cgh.parallel_for<detail::test_kernel<__LINE__, int>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              size_t gid = item.get_global_linear_id();
              size_t lid = g.get_local_linear_id();
              int tmp = 0;

              for (int i = 0; i < local_size; ++i) {
                if (lid == i) {
                  out_acc[gid] = tmp;
                  tmp++;
                }
                sycl::group_barrier(g);
                if (lid == i)
                  scratch[0] = tmp;
                sycl::group_barrier(g);
                tmp = scratch[0];
              }
            });
      });
    }

    for (size_t gid = 0; gid < num_groups; ++gid) {
      for (size_t lid = 0; lid < local_size; ++lid) {
        int computed = out_data[gid * local_size + lid];
        int expected = lid;
        BOOST_TEST(detail::compare_type(computed, expected),
            detail::type_to_string(computed)
                << " at position " << lid << " (group: " << gid << ", local size: " << local_size
                << " instead of " << detail::type_to_string(expected));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_broadcast, T, test_types) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> out_data(global_size);

    {
      sycl::buffer<T, 1> out_buf{out_data.data(), out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor out_acc{out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              size_t gid = item.get_global_linear_id();

              out_acc[gid] = sycl::group_broadcast(
                  g, static_cast<T>(
                         static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                             gid)));
            });
      });
    }

    for (size_t gid = 0; gid < num_groups; ++gid) {
      T expected =
          static_cast<T>(static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
              gid * local_size));
      for (size_t lid = 0; lid < local_size; ++lid) {
        T computed = out_data[gid * local_size + lid];
        BOOST_TEST(detail::compare_type(computed, expected),
            detail::type_to_string(computed)
                << " at position " << lid << " (group: " << gid << ", local size: " << local_size
                << " instead of " << detail::type_to_string(expected));
      }
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_sub_group_broadcast, T, test_types) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> out_data(global_size);

    {
      sycl::buffer<T, 1> out_buf{out_data.data(), out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor out_acc{out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto sg = item.get_sub_group();
              size_t gid = item.get_global_linear_id();

              out_acc[gid] = sycl::group_broadcast(
                  sg, static_cast<T>(
                          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                              gid)));
            });
      });
    }

    auto sub_group_size = std::min(
        (size_t)sycl::queue{}.get_device().get_info<sycl::info::device::sub_group_sizes>()[0],
        local_size);
    size_t gid = 0;
    for (size_t lid = 0; lid < local_size; ++lid) {
      T expected =
          static_cast<T>(static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
              lid / sub_group_size));
      T computed = out_data[gid * local_size + lid];
      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed)
              << " at position " << lid << " (group: " << gid << ", local size: " << local_size
              << " instead of " << detail::type_to_string(expected));
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_sub_group_shuffle_like, T, test_types) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> left_out_data(global_size);
    std::vector<T> right_out_data(global_size);
    std::vector<T> xor_out_data(global_size);
    std::vector<T> select_out_data(global_size);

    {
      sycl::buffer<T, 1> left_out_buf{left_out_data.data(), left_out_data.size()};
      sycl::buffer<T, 1> right_out_buf{right_out_data.data(), right_out_data.size()};
      sycl::buffer<T, 1> xor_out_buf{xor_out_data.data(), xor_out_data.size()};
      sycl::buffer<T, 1> select_out_buf{select_out_data.data(), select_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor left_out_acc{left_out_buf, cgh, sycl::write_only};
        sycl::accessor right_out_acc{right_out_buf, cgh, sycl::write_only};
        sycl::accessor xor_out_acc{xor_out_buf, cgh, sycl::write_only};
        sycl::accessor select_out_acc{select_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto sg = item.get_sub_group();
              size_t gid = item.get_global_linear_id();

              left_out_acc[gid] = sycl::shift_group_left(sg,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                          gid)),
                  1);
              right_out_acc[gid] = sycl::shift_group_right(sg,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                          gid)),
                  1);
              xor_out_acc[gid] = sycl::permute_group_by_xor(sg,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                          gid)),
                  0x01);
              select_out_acc[gid] = sycl::select_from_group(sg,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                          gid)),
                  0);
            });
      });
    }

    auto sub_group_size = std::min(
        (size_t)sycl::queue{}.get_device().get_info<sycl::info::device::sub_group_sizes>()[0],
        local_size);

    // shift_group_left last element is unspecified
    for (size_t sg_lid = 0; sg_lid < sub_group_size - 1; ++sg_lid) {
      T computed = left_out_data[sg_lid];
      T expected = static_cast<T>(
          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(sg_lid + 1));
      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed)
              << " at position " << sg_lid << " (variant: left, local size: " << local_size
              << " instead of " << detail::type_to_string(expected));
    }

    // shift_group_right first element is unspecified
    for (size_t sg_lid = 1; sg_lid < sub_group_size; ++sg_lid) {
      T computed = right_out_data[sg_lid];
      T expected = static_cast<T>(
          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(sg_lid - 1));
      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed)
              << " at position " << sg_lid << " (variant: right, local size: " << local_size
              << " instead of " << detail::type_to_string(expected));
    }

    // permute_group_by_xor some elements are unspecified
    for (size_t sg_lid = 0; sg_lid < sub_group_size; ++sg_lid) {
      if ((sg_lid ^ 0x01) >= sub_group_size) // check if unspecified
        continue;

      T computed = xor_out_data[sg_lid];
      T expected = static_cast<T>(
          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(sg_lid ^ 0x01));
      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed)
              << " at position " << sg_lid << " (variant: xor, local size: " << local_size
              << " instead of " << detail::type_to_string(expected));
    }

    // select_from_group
    for (size_t sg_lid = 0; sg_lid < sub_group_size; ++sg_lid) {
      T computed = select_out_data[sg_lid];
      T expected = static_cast<T>(
          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0));
      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed)
              << " at position " << sg_lid << " (variant: select, local size: " << local_size
              << " instead of " << detail::type_to_string(expected));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
