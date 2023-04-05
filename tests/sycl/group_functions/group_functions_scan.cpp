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

#include <cstddef>
#include <functional>

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"
#include <boost/test/data/test_case.hpp>

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

using namespace cl;

namespace detail {
template<typename T>
std::vector<T> generate_input_data_scan(size_t local_size, int num_groups) {
  std::vector<T> data(local_size * num_groups);

  // each group gets 2 as first element and 1 for the rest
  for (size_t i = 0; i < num_groups; ++i) {
    data[i * local_size] =
        static_cast<T>(static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(2));
    for (size_t j = 1; j < local_size; ++j)
      data[i * local_size + j] = static_cast<T>(
          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1));
  }

  return data;
}

template<typename T>
void check_output_group_data_scan(std::vector<T> in_data, std::vector<T> out_data,
    size_t local_size, size_t num_groups, int test_case) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size;
    std::vector<T> expected(in_data.size());
    auto start = in_data.begin() + i * local_size;
    auto end = start + num_elements;
    if (test_case == 0) {
      std::inclusive_scan(start, end, expected.begin(), std::plus<T>(),
          static_cast<T>(static_cast<element_type>(0)));
    } else if (test_case == 1) {
      std::inclusive_scan(start, end, expected.begin(), std::multiplies<T>(),
          static_cast<T>(static_cast<element_type>(1)));
    } else if (test_case == 2) {
      std::exclusive_scan(start, end, expected.begin(),
          static_cast<T>(static_cast<element_type>(0)), std::plus<T>());
    } else {
      std::exclusive_scan(start, end, expected.begin(),
          static_cast<T>(static_cast<element_type>(1)), std::multiplies<T>());
    }
    for (int lid = 0; lid < local_size; ++lid) {
      int gid = lid + i * local_size;
      T computed = out_data[gid];
      T expected_element = expected[lid];

      BOOST_TEST(detail::compare_type(computed, expected_element),
          detail::type_to_string(computed)
              << " at position " << lid << " (group: " << i << ", gid: " << gid
              << ", local size: " << local_size << ", case: " << test_case
              << "[0=inclusive plus, 1=inclusive multiply, 2=exclusive plus, 3=exclusive multiply])"
              << " instead of " << detail::type_to_string(expected_element));
    }
  }
}

template<typename T>
void check_output_joint_data_scan(std::vector<T> in_data, std::vector<T> out_data,
    size_t local_size, size_t num_groups, size_t elements_per_item, int test_case) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size * elements_per_item;
    std::vector<T> expected(in_data.size());
    auto start = in_data.begin() + i * local_size * elements_per_item;
    auto end = start + num_elements;
    if (test_case == 0) {
      std::inclusive_scan(start, end, expected.begin(), std::plus<T>(),
          static_cast<T>(static_cast<element_type>(0)));
    } else if (test_case == 1) {
      std::inclusive_scan(start, end, expected.begin(), std::multiplies<T>(),
          static_cast<T>(static_cast<element_type>(1)));
    } else if (test_case == 2) {
      std::exclusive_scan(start, end, expected.begin(),
          static_cast<T>(static_cast<element_type>(0)), std::plus<T>());
    } else {
      std::exclusive_scan(start, end, expected.begin(),
          static_cast<T>(static_cast<element_type>(1)), std::multiplies<T>());
    }
    for (int lid = 0; lid < num_elements; ++lid) {
      int gid = lid + i * num_elements;
      typename sycl::detail::builtin_type_traits<T>::element_type computed =
          sycl::detail::builtin_type_traits<T>::element(out_data[gid], 0);
      element_type expected_element =
          sycl::detail::builtin_type_traits<T>::element(expected[lid], 0);

      BOOST_TEST(detail::compare_type(computed, expected_element),
          detail::type_to_string(computed)
              << " at position " << lid << " (group: " << i << ", lid: " << lid
              << ", local size: " << local_size << ", case: " << test_case
              << "[0=inclusive plus, 1=inclusive multiply, 2=exclusive plus, 3=exclusive multiply])"
              << " instead of " << detail::type_to_string(expected_element));
    }
  }
}

template<int Line, typename T>
class test_kernel;
} // namespace detail

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

/*
 * ND-range
 */

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_scan_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_scan<T>(local_size, global_size);
    std::vector<T> plus_in_out_data(global_size);
    std::vector<T> mult_in_out_data(global_size);
    std::vector<T> plus_ex_out_data(global_size);
    std::vector<T> mult_ex_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_in_out_buf{plus_in_out_data.data(), plus_in_out_data.size()};
      sycl::buffer<T, 1> mult_in_out_buf{mult_in_out_data.data(), mult_in_out_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_in_out_acc{plus_in_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_in_out_acc{mult_in_out_buf, cgh, sycl::write_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              size_t gid = item.get_global_linear_id();

              plus_in_out_acc[gid] = inclusive_scan_over_group(g, in_acc[gid], std::plus<T>());
              mult_in_out_acc[gid] =
                  inclusive_scan_over_group(g, in_acc[gid], std::multiplies<T>());
              plus_ex_out_acc[gid] = exclusive_scan_over_group(g, in_acc[gid],
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
                  std::plus<T>());
              mult_ex_out_acc[gid] = exclusive_scan_over_group(g, in_acc[gid],
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
                  std::multiplies<T>());
            });
      });
    }
    detail::check_output_group_data_scan(in_data, plus_in_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_scan(in_data, mult_in_out_data, local_size, num_groups, 1);
    detail::check_output_group_data_scan(in_data, plus_ex_out_data, local_size, num_groups, 2);
    detail::check_output_group_data_scan(in_data, mult_ex_out_data, local_size, num_groups, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_subgroup_x_scan_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_scan<T>(local_size, global_size);
    std::vector<T> plus_in_out_data(global_size);
    std::vector<T> mult_in_out_data(global_size);
    std::vector<T> plus_ex_out_data(global_size);
    std::vector<T> mult_ex_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_in_out_buf{plus_in_out_data.data(), plus_in_out_data.size()};
      sycl::buffer<T, 1> mult_in_out_buf{mult_in_out_data.data(), mult_in_out_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_in_out_acc{plus_in_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_in_out_acc{mult_in_out_buf, cgh, sycl::write_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_sub_group();
              size_t gid = item.get_global_linear_id();

              plus_in_out_acc[gid] = inclusive_scan_over_group(g, in_acc[gid], std::plus<T>());
              mult_in_out_acc[gid] =
                  inclusive_scan_over_group(g, in_acc[gid], std::multiplies<T>());
              plus_ex_out_acc[gid] = exclusive_scan_over_group(g, in_acc[gid],
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
                  std::plus<T>());
              mult_ex_out_acc[gid] = exclusive_scan_over_group(g, in_acc[gid],
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
                  std::multiplies<T>());
            });
      });
    }
    detail::check_output_group_data_scan(in_data, plus_in_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_scan(in_data, mult_in_out_data, local_size, num_groups, 1);
    detail::check_output_group_data_scan(in_data, plus_ex_out_data, local_size, num_groups, 2);
    detail::check_output_group_data_scan(in_data, mult_ex_out_data, local_size, num_groups, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_joint_scan_init, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_scan<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_ex_out_data(global_size * elements_per_item);
    std::vector<T> mult_ex_out_data(global_size * elements_per_item);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;
              auto plus_ex_out_start =
                  plus_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto mult_ex_out_start =
                  mult_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;

              // only write back the first element of a vectortype
              sycl::joint_exclusive_scan(g, start, end, plus_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
                  std::plus<T>());
              sycl::joint_exclusive_scan(g, start, end, mult_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
                  std::multiplies<T>());
            });
      });
    }
    detail::check_output_joint_data_scan(
        in_data, plus_ex_out_data, local_size, num_groups, elements_per_item, 2);
    detail::check_output_joint_data_scan(
        in_data, mult_ex_out_data, local_size, num_groups, elements_per_item, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_subgroup_joint_scan_init, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }

  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_scan<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_ex_out_data(global_size * elements_per_item);
    std::vector<T> mult_ex_out_data(global_size * elements_per_item);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto sg = item.get_sub_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;
              auto plus_ex_out_start =
                  plus_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto mult_ex_out_start =
                  mult_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;

              // only write back the first element of a vectortype
              sycl::joint_exclusive_scan(g, start, end, plus_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
                  std::plus<T>());
              sycl::joint_exclusive_scan(g, start, end, mult_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
                  std::multiplies<T>());
            });
      });
    }
    detail::check_output_joint_data_scan(
        in_data, plus_ex_out_data, local_size, num_groups, elements_per_item, 2);
    detail::check_output_joint_data_scan(
        in_data, mult_ex_out_data, local_size, num_groups, elements_per_item, 3);
  }
}

/*
 * scoped V2
 */
#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX
BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_group_scan_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_scan<T>(local_size, global_size);
    std::vector<T> plus_in_out_data(global_size);
    std::vector<T> mult_in_out_data(global_size);
    std::vector<T> plus_ex_out_data(global_size);
    std::vector<T> mult_ex_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_in_out_buf{plus_in_out_data.data(), plus_in_out_data.size()};
      sycl::buffer<T, 1> mult_in_out_buf{mult_in_out_data.data(), mult_in_out_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_in_out_acc{plus_in_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_in_out_acc{mult_in_out_buf, cgh, sycl::write_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__,
            T>>(sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
          sycl::memory_environment(g, sycl::require_private_mem<T>(),
              sycl::require_private_mem<T>(), sycl::require_private_mem<T>(),
              sycl::require_private_mem<T>(), sycl::require_private_mem<T>(),
              [&](auto &private_mem, auto &private_mem_plus_in, auto &private_mem_mult_in,
                  auto &private_mem_plus_ex, auto &private_mem_mult_ex) {
                // load values into private memory
                sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                  private_mem(idx) = in_acc[idx.get_global_linear_id()];
                });

                sycl::inclusive_scan_over_group(
                    g, private_mem, std::plus<T>(), private_mem_plus_in);
                sycl::inclusive_scan_over_group(
                    g, private_mem, std::multiplies<T>(), private_mem_mult_in);
                sycl::exclusive_scan_over_group(g, private_mem,
                    static_cast<T>(
                        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                            0)),
                    std::plus<T>(), private_mem_plus_ex);
                sycl::exclusive_scan_over_group(g, private_mem,
                    static_cast<T>(
                        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                            1)),
                    std::multiplies<T>(), private_mem_mult_ex);

                // store values into global memory
                sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                  plus_in_out_acc[idx.get_global_linear_id()] = private_mem_plus_in(idx);
                  mult_in_out_acc[idx.get_global_linear_id()] = private_mem_mult_in(idx);
                  plus_ex_out_acc[idx.get_global_linear_id()] = private_mem_plus_ex(idx);
                  mult_ex_out_acc[idx.get_global_linear_id()] = private_mem_mult_ex(idx);
                });
              });
        });
      });
    }
    detail::check_output_group_data_scan(in_data, plus_in_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_scan(in_data, mult_in_out_data, local_size, num_groups, 1);
    detail::check_output_group_data_scan(in_data, plus_ex_out_data, local_size, num_groups, 2);
    detail::check_output_group_data_scan(in_data, mult_ex_out_data, local_size, num_groups, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_subgroup_x_scan_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_scan<T>(local_size, global_size);
    std::vector<T> plus_in_out_data(global_size);
    std::vector<T> mult_in_out_data(global_size);
    std::vector<T> plus_ex_out_data(global_size);
    std::vector<T> mult_ex_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_in_out_buf{plus_in_out_data.data(), plus_in_out_data.size()};
      sycl::buffer<T, 1> mult_in_out_buf{mult_in_out_data.data(), mult_in_out_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_in_out_acc{plus_in_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_in_out_acc{mult_in_out_buf, cgh, sycl::write_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__,
            T>>(sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
          sycl::distribute_groups(g, [&](auto sg) {
            sycl::memory_environment(sg, sycl::require_private_mem<T>(),
                sycl::require_private_mem<T>(), sycl::require_private_mem<T>(),
                sycl::require_private_mem<T>(), sycl::require_private_mem<T>(),
                [&](auto &private_mem, auto &private_mem_plus_in, auto &private_mem_mult_in,
                    auto &private_mem_plus_ex, auto &private_mem_mult_ex) {
                  // load values into private memory
                  sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                    private_mem(idx) = in_acc[idx.get_global_linear_id()];
                  });

                  sycl::inclusive_scan_over_group(
                      sg, private_mem, std::plus<T>(), private_mem_plus_in);
                  sycl::inclusive_scan_over_group(
                      sg, private_mem, std::multiplies<T>(), private_mem_mult_in);
                  sycl::exclusive_scan_over_group(sg, private_mem,
                      static_cast<T>(
                          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                              0)),
                      std::plus<T>(), private_mem_plus_ex);
                  sycl::exclusive_scan_over_group(sg, private_mem,
                      static_cast<T>(
                          static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                              1)),
                      std::multiplies<T>(), private_mem_mult_ex);

                  // store values into global memory
                  sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                    plus_in_out_acc[idx.get_global_linear_id()] = private_mem_plus_in(idx);
                    mult_in_out_acc[idx.get_global_linear_id()] = private_mem_mult_in(idx);
                    plus_ex_out_acc[idx.get_global_linear_id()] = private_mem_plus_ex(idx);
                    mult_ex_out_acc[idx.get_global_linear_id()] = private_mem_mult_ex(idx);
                  });
                });
          });
        });
      });
    }
    detail::check_output_group_data_scan(in_data, plus_in_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_scan(in_data, mult_in_out_data, local_size, num_groups, 1);
    detail::check_output_group_data_scan(in_data, plus_ex_out_data, local_size, num_groups, 2);
    detail::check_output_group_data_scan(in_data, mult_ex_out_data, local_size, num_groups, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_group_joint_scan_init, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_scan<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_ex_out_data(global_size * elements_per_item);
    std::vector<T> mult_ex_out_data(global_size * elements_per_item);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;
              auto plus_ex_out_start =
                  plus_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto mult_ex_out_start =
                  mult_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;

              // only write back the first element of a vectortype
              sycl::joint_exclusive_scan(g, start, end, plus_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
                  std::plus<T>());
              sycl::joint_exclusive_scan(g, start, end, mult_ex_out_start,
                  static_cast<T>(
                      static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
                  std::multiplies<T>());
            });
      });
    }
    detail::check_output_joint_data_scan(
        in_data, plus_ex_out_data, local_size, num_groups, elements_per_item, 2);
    detail::check_output_joint_data_scan(
        in_data, mult_ex_out_data, local_size, num_groups, elements_per_item, 3);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_subgroup_joint_scan_init, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }

  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_scan<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_ex_out_data(global_size * elements_per_item);
    std::vector<T> mult_ex_out_data(global_size * elements_per_item);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_ex_out_buf{plus_ex_out_data.data(), plus_ex_out_data.size()};
      sycl::buffer<T, 1> mult_ex_out_buf{mult_ex_out_data.data(), mult_ex_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_ex_out_acc{plus_ex_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_ex_out_acc{mult_ex_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;
              auto plus_ex_out_start =
                  plus_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto mult_ex_out_start =
                  mult_ex_out_acc.get_pointer().get() + group_id * local_size * elements_per_item;

              sycl::distribute_groups(g, [&](auto sg) {
                // only write back the first element of a vectortype
                sycl::joint_exclusive_scan(sg, start, end, plus_ex_out_start,
                    static_cast<T>(
                        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                            0)),
                    std::plus<T>());
                sycl::joint_exclusive_scan(sg, start, end, mult_ex_out_start,
                    static_cast<T>(
                        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(
                            1)),
                    std::multiplies<T>());
              });
            });
      });
    }
    detail::check_output_joint_data_scan(
        in_data, plus_ex_out_data, local_size, num_groups, elements_per_item, 2);
    detail::check_output_joint_data_scan(
        in_data, mult_ex_out_data, local_size, num_groups, elements_per_item, 3);
  }
}
#endif

BOOST_AUTO_TEST_SUITE_END()

#endif
