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
std::vector<T> generate_input_data_binary_reduce(size_t local_size, size_t global_size) {
  std::vector<T> data(global_size);

  for (size_t i = 0; i < 2 * local_size; ++i)
    data[i] = static_cast<T>(
        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(false));
  for (size_t i = 2 * local_size; i < global_size; ++i)
    data[i] = static_cast<T>(
        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(true));

  if (local_size > 1) {
    // add elements to generate ll possible 4 cases (all true/false, all but 1 true/false)
    data[static_cast<size_t>(local_size / 2)] = static_cast<T>(
        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(true));
    data[2 * local_size + static_cast<size_t>(local_size / 2)] = static_cast<T>(
        static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(false));
  }

  return data;
}

template<typename T>
void check_output_group_data_binary_reduce(std::vector<T> in_data, std::vector<char> out_data,
    size_t local_size, size_t global_size, int test_case) {
  constexpr size_t num_groups = 4;

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size;
    bool expected;
    auto start = in_data.begin() + i * local_size;
    auto end = start + num_elements;
    if (test_case == 0) {
      expected = std::any_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    } else if (test_case == 1) {
      expected = std::all_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    } else {
      expected = std::none_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    }
    for (int lid = 0; lid < num_elements; ++lid) {
      int gid = lid + i * local_size;
      bool computed = out_data[gid];

      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed) << " at position " << lid << " (group: " << i
                                           << ", gid: " << gid << ", local size: " << local_size
                                           << ", case: " << test_case << "[0=any, 1=all, 2=none])"
                                           << " instead of " << detail::type_to_string(expected));
    }
  }
}

template<typename T>
void check_output_joint_data_binary_reduce(std::vector<T> in_data, std::vector<char> out_data,
    size_t local_size, size_t global_size, size_t elements_per_item, int test_case) {
  constexpr size_t num_groups = 4;

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size * elements_per_item;
    bool expected;
    auto start = in_data.begin() + i * local_size * elements_per_item;
    auto end = start + num_elements;
    if (test_case == 0) {
      expected = std::any_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    } else if (test_case == 1) {
      expected = std::all_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    } else {
      expected = std::none_of(start, end, [](T x) {
        return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
      });
    }
    for (int lid = 0; lid < local_size; ++lid) {
      int gid = lid + i * local_size;
      bool computed = out_data[gid];

      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed) << " at position " << lid << " (group: " << i
                                           << ", gid: " << gid << ", local size: " << local_size
                                           << ", case: " << test_case << "[0=any, 1=all, 2=none])"
                                           << " instead of " << detail::type_to_string(expected));
    }
  }
}

} // namespace detail

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

/*
 * ND-range
 */

BOOST_AUTO_TEST_CASE(nd_range_group_x_of_group) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<char> in_data =
        detail::generate_input_data_binary_reduce<char>(local_size, global_size);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<char, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, char>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              size_t gid = item.get_global_linear_id();

              any_out_acc[gid] = any_of_group(g, in_acc[gid]);
              all_out_acc[gid] = all_of_group(g, in_acc[gid]);
              none_out_acc[gid] = none_of_group(g, in_acc[gid]);
            });
      });
    }
    detail::check_output_group_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, 0);
    detail::check_output_group_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, 1);
    detail::check_output_group_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, 2);
  }
}

BOOST_AUTO_TEST_CASE(nd_range_subgroup_x_of_group) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<char> in_data =
        detail::generate_input_data_binary_reduce<char>(local_size, global_size);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<char, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, char>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_sub_group();
              size_t gid = item.get_global_linear_id();

              any_out_acc[gid] = any_of_group(g, in_acc[gid]);
              all_out_acc[gid] = all_of_group(g, in_acc[gid]);
              none_out_acc[gid] = none_of_group(g, in_acc[gid]);
            });
      });
    }
    detail::check_output_group_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, 0);
    detail::check_output_group_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, 1);
    detail::check_output_group_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, 2);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_joint_x_of, T, test_types) {
  constexpr size_t num_groups = 4;        // needed for correct data generation
  constexpr size_t elements_per_item = 4; // needed for correct data generation

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<T> in_data = detail::generate_input_data_binary_reduce<T>(
        local_size * elements_per_item, global_size * elements_per_item);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              any_out_acc[gid] = joint_any_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              all_out_acc[gid] = joint_all_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              none_out_acc[gid] = joint_none_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
            });
      });
    }
    detail::check_output_joint_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, elements_per_item, 0);
    detail::check_output_joint_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, elements_per_item, 1);
    detail::check_output_joint_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, elements_per_item, 2);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_subgroup_joint_x_of, T, test_types) {
  constexpr size_t num_groups = 4;        // needed for correct data generation
  constexpr size_t elements_per_item = 4; // needed for correct data generation

  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<T> in_data = detail::generate_input_data_binary_reduce<T>(
        local_size * elements_per_item, global_size * elements_per_item);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto sg = item.get_sub_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              any_out_acc[gid] = joint_any_of(sg, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              all_out_acc[gid] = joint_all_of(sg, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              none_out_acc[gid] = joint_none_of(sg, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
            });
      });
    }
    detail::check_output_joint_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, elements_per_item, 0);
    detail::check_output_joint_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, elements_per_item, 1);
    detail::check_output_joint_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, elements_per_item, 2);
  }
}


/*
 * scoped V2
 */
#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX
BOOST_AUTO_TEST_CASE(scopedv2_group_x_of_group) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<char> in_data =
        detail::generate_input_data_binary_reduce<char>(local_size, global_size);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<char, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};


        cgh.parallel<detail::test_kernel<__LINE__, char>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              sycl::memory_environment(
                  g, sycl::require_private_mem<bool>(), [&](auto &private_mem) {
                    // load values into private memory
                    sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                      private_mem(idx) = in_acc[idx.get_global_linear_id()];
                    });

                    bool any_out = sycl::any_of_group(g, private_mem);
                    bool all_out = sycl::all_of_group(g, private_mem);
                    bool none_out = sycl::none_of_group(g, private_mem);

                    // write result back to global memory
                    sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                      any_out_acc[idx.get_global_linear_id()] = any_out;
                      all_out_acc[idx.get_global_linear_id()] = all_out;
                      none_out_acc[idx.get_global_linear_id()] = none_out;
                    });
                  });
            });
      });
    }
    detail::check_output_group_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, 0);
    detail::check_output_group_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, 1);
    detail::check_output_group_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, 2);
  }
}

BOOST_AUTO_TEST_CASE(scopedv2_subgroup_x_of_group) {
  constexpr size_t num_groups = 4; // needed for correct data generation
  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<char> in_data =
        detail::generate_input_data_binary_reduce<char>(local_size, global_size);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<char, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, char>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              bool any_out;
              bool all_out;
              bool none_out;
              sycl::distribute_groups(g, [&](auto sg) {
                sycl::memory_environment(
                    sg, sycl::require_private_mem<bool>(), [&](auto &private_mem) {
                      // load values into private memory
                      sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                        private_mem(idx) = in_acc[idx.get_global_linear_id()];
                      });

                      any_out = sycl::any_of_group(sg, private_mem);
                      all_out = sycl::all_of_group(sg, private_mem);
                      none_out = sycl::none_of_group(sg, private_mem);
                    });
              });

              // write result back to global memory
              sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                any_out_acc[idx.get_global_linear_id()] = any_out;
                all_out_acc[idx.get_global_linear_id()] = all_out;
                none_out_acc[idx.get_global_linear_id()] = none_out;
              });
            });
      });
    }
    detail::check_output_group_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, 0);
    detail::check_output_group_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, 1);
    detail::check_output_group_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, 2);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_group_joint_x_of, T, test_types) {
  constexpr size_t num_groups = 4;        // needed for correct data generation
  constexpr size_t elements_per_item = 4; // needed for correct data generation

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<T> in_data = detail::generate_input_data_binary_reduce<T>(
        local_size * elements_per_item, global_size * elements_per_item);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              bool any_out = sycl::joint_any_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              bool all_out = sycl::joint_all_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });
              bool none_out = sycl::joint_none_of(g, start, end, [](T x) {
                return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
              });

              // write result back to global memory
              sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                any_out_acc[idx.get_global_linear_id()] = any_out;
                all_out_acc[idx.get_global_linear_id()] = all_out;
                none_out_acc[idx.get_global_linear_id()] = none_out;
              });
            });
      });
    }
    detail::check_output_joint_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, elements_per_item, 0);
    detail::check_output_joint_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, elements_per_item, 1);
    detail::check_output_joint_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, elements_per_item, 2);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_subgroup_joint_x_of, T, test_types) {
  constexpr size_t num_groups = 4;        // needed for correct data generation
  constexpr size_t elements_per_item = 4; // needed for correct data generation

  std::vector<size_t> local_sizes = {1};
  auto device = sycl::queue{}.get_device();
  if (!device.is_host()) {
    local_sizes.push_back(device.get_info<sycl::info::device::sub_group_sizes>()[0]);
  }
  for (const size_t &local_size : local_sizes) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    // using char instead of bool for result, because of vector<bool> specialization
    std::vector<T> in_data = detail::generate_input_data_binary_reduce<T>(
        local_size * elements_per_item, global_size * elements_per_item);
    std::vector<char> any_out_data(global_size);
    std::vector<char> all_out_data(global_size);
    std::vector<char> none_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<char, 1> any_out_buf{any_out_data.data(), any_out_data.size()};
      sycl::buffer<char, 1> all_out_buf{all_out_data.data(), all_out_data.size()};
      sycl::buffer<char, 1> none_out_buf{none_out_data.data(), none_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor any_out_acc{any_out_buf, cgh, sycl::write_only};
        sycl::accessor all_out_acc{all_out_buf, cgh, sycl::write_only};
        sycl::accessor none_out_acc{none_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              bool any_out;
              bool all_out;
              bool none_out;
              sycl::distribute_groups(g, [&](auto sg) {
                any_out = sycl::joint_any_of(g, start, end, [](T x) {
                  return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
                });
                all_out = sycl::joint_all_of(g, start, end, [](T x) {
                  return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
                });
                none_out = sycl::joint_none_of(g, start, end, [](T x) {
                  return static_cast<bool>(sycl::detail::builtin_type_traits<T>::element(x, 0));
                });
              });

              // write result back to global memory
              sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                any_out_acc[idx.get_global_linear_id()] = any_out;
                all_out_acc[idx.get_global_linear_id()] = all_out;
                none_out_acc[idx.get_global_linear_id()] = none_out;
              });
            });
      });
    }
    detail::check_output_joint_data_binary_reduce(
        in_data, any_out_data, local_size, global_size, elements_per_item, 0);
    detail::check_output_joint_data_binary_reduce(
        in_data, all_out_data, local_size, global_size, elements_per_item, 1);
    detail::check_output_joint_data_binary_reduce(
        in_data, none_out_data, local_size, global_size, elements_per_item, 2);
  }
}
#endif

BOOST_AUTO_TEST_SUITE_END()

#endif
