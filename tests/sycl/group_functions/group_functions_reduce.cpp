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
std::vector<T> generate_input_data_reduce(size_t local_size, int num_groups) {
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
void check_output_group_data_reduce(std::vector<T> in_data, std::vector<T> out_data,
    size_t local_size, size_t num_groups, int test_case) {

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size;
    T expected;
    auto start = in_data.begin() + i * local_size;
    auto end = start + num_elements;
    if (test_case == 0) {
      expected = std::reduce(start, end,
          static_cast<T>(
              static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
          [](T x, T y) { return x + y; });
    } else {
      expected = std::reduce(start, end,
          static_cast<T>(
              static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
          [](T x, T y) { return x * y; });
    }
    for (int lid = 0; lid < num_elements; ++lid) {
      int gid = lid + i * local_size;
      T computed = out_data[gid];

      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed) << " at position " << lid << " (group: " << i
                                           << ", gid: " << gid << ", local size: " << local_size
                                           << ", case: " << test_case << "[0=plus, 1=multiply])"
                                           << " instead of " << detail::type_to_string(expected));
    }
  }
}

template<typename T>
void check_output_joint_data_reduce(std::vector<T> in_data, std::vector<T> out_data,
    size_t local_size, size_t num_groups, size_t elements_per_item, int test_case) {

  for (int i = 0; i < num_groups; ++i) {
    size_t num_elements = local_size * elements_per_item;
    T expected;
    auto start = in_data.begin() + i * local_size * elements_per_item;
    auto end = start + num_elements;
    if (test_case == 0) {
      expected = std::reduce(start, end,
          static_cast<T>(
              static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(0)),
          [](T x, T y) { return x + y; });
    } else {
      expected = std::reduce(start, end,
          static_cast<T>(
              static_cast<typename sycl::detail::builtin_type_traits<T>::element_type>(1)),
          [](T x, T y) { return x * y; });
    }
    for (int lid = 0; lid < local_size; ++lid) {
      int gid = lid + i * local_size;
      T computed = out_data[gid];

      BOOST_TEST(detail::compare_type(computed, expected),
          detail::type_to_string(computed) << " at position " << lid << " (group: " << i
                                           << ", gid: " << gid << ", local size: " << local_size
                                           << ", case: " << test_case << "[0=plus, 1=multiply])"
                                           << " instead of " << detail::type_to_string(expected));
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

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_reduce_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_reduce<T>(local_size, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              size_t gid = item.get_global_linear_id();

              plus_out_acc[gid] = reduce_over_group(g, in_acc[gid], std::plus<T>());
              mult_out_acc[gid] = reduce_over_group(g, in_acc[gid], std::multiplies<T>());
            });
      });
    }
    detail::check_output_group_data_reduce(in_data, plus_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_reduce(in_data, mult_out_data, local_size, num_groups, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_subgroup_x_reduce_over_group, T, test_types) {
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

    std::vector<T> in_data = detail::generate_input_data_reduce<T>(local_size, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_sub_group();
              size_t gid = item.get_global_linear_id();

              plus_out_acc[gid] = reduce_over_group(g, in_acc[gid], std::plus<T>());
              mult_out_acc[gid] = reduce_over_group(g, in_acc[gid], std::multiplies<T>());
            });
      });
    }
    detail::check_output_group_data_reduce(in_data, plus_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_reduce(in_data, mult_out_data, local_size, num_groups, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_group_joint_reduce, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_reduce<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              // only write back the first element of a vectortype
              plus_out_acc[gid] = joint_reduce(g, start, end, std::plus<T>());
              mult_out_acc[gid] = joint_reduce(g, start, end, std::multiplies<T>());
            });
      });
    }
    detail::check_output_joint_data_reduce(
        in_data, plus_out_data, local_size, num_groups, elements_per_item, 0);
    detail::check_output_joint_data_reduce(
        in_data, mult_out_data, local_size, num_groups, elements_per_item, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(nd_range_subgroup_joint_reduce, T, test_types) {
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
        detail::generate_input_data_reduce<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel_for<detail::test_kernel<__LINE__, T>>(
            sycl::nd_range<1>{global_size, local_size}, [=](sycl::nd_item<1> item) {
              auto g = item.get_group();
              auto sg = item.get_sub_group();
              auto group_id = g.get_group_linear_id();
              size_t gid = item.get_global_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              // only write back the first element of a vectortype
              plus_out_acc[gid] = joint_reduce(sg, start, end, std::plus<T>());
              mult_out_acc[gid] = joint_reduce(sg, start, end, std::multiplies<T>());
            });
      });
    }
    detail::check_output_joint_data_reduce(
        in_data, plus_out_data, local_size, num_groups, elements_per_item, 0);
    detail::check_output_joint_data_reduce(
        in_data, mult_out_data, local_size, num_groups, elements_per_item, 1);
  }
}

/*
 * scoped V2
 */

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_group_reduce_over_group, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data = detail::generate_input_data_reduce<T>(local_size, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              sycl::memory_environment(g, sycl::require_private_mem<T>(), [&](auto &private_mem) {
                // load values into private memory
                sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                  private_mem(idx) = in_acc[idx.get_global_linear_id()];
                });

                T plus_result = sycl::reduce_over_group(g, private_mem, std::plus<T>());
                T mult_result = sycl::reduce_over_group(g, private_mem, std::multiplies<T>());

                // store values into global memory
                sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                  plus_out_acc[idx.get_global_linear_id()] = plus_result;
                  mult_out_acc[idx.get_global_linear_id()] = mult_result;
                });
              });
            });
      });
    }
    detail::check_output_group_data_reduce(in_data, plus_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_reduce(in_data, mult_out_data, local_size, num_groups, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_subgroup_x_reduce_over_group, T, test_types) {
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

    std::vector<T> in_data = detail::generate_input_data_reduce<T>(local_size, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              sycl::distribute_groups(g, [&](auto sg) {
                sycl::memory_environment(
                    sg, sycl::require_private_mem<T>(), [&](auto &private_mem) {
                      // load values into private memory
                      sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                        private_mem(idx) = in_acc[idx.get_global_linear_id()];
                      });

                      T plus_result = sycl::reduce_over_group(sg, private_mem, std::plus<T>());
                      T mult_result =
                          sycl::reduce_over_group(sg, private_mem, std::multiplies<T>());

                      // store values into global memory
                      sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                        plus_out_acc[idx.get_global_linear_id()] = plus_result;
                        mult_out_acc[idx.get_global_linear_id()] = mult_result;
                      });
                    });
              });
            });
      });
    }
    detail::check_output_group_data_reduce(in_data, plus_out_data, local_size, num_groups, 0);
    detail::check_output_group_data_reduce(in_data, mult_out_data, local_size, num_groups, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_group_joint_reduce, T, test_types) {
  using element_type = typename sycl::detail::builtin_type_traits<T>::element_type;
  constexpr size_t num_groups = 4;
  constexpr size_t elements_per_item = 4;

  for (const size_t &local_size : {25, 32, 64, 128, 256, 500, 512, 1024}) {
    // used to generate incomplete groups for nd-range
    const size_t global_size = num_groups * local_size;

    std::vector<T> in_data =
        detail::generate_input_data_reduce<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              T plus_result = sycl::joint_reduce(g, start, end, std::plus<T>());
              T mult_result = sycl::joint_reduce(g, start, end, std::multiplies<T>());

              // store values into global memory
              sycl::distribute_items(g, [&](sycl::s_item<1> idx) {
                plus_out_acc[idx.get_global_linear_id()] = plus_result;
                mult_out_acc[idx.get_global_linear_id()] = mult_result;
              });
            });
      });
    }
    detail::check_output_joint_data_reduce(
        in_data, plus_out_data, local_size, num_groups, elements_per_item, 0);
    detail::check_output_joint_data_reduce(
        in_data, mult_out_data, local_size, num_groups, elements_per_item, 1);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(scopedv2_subgroup_joint_reduce, T, test_types) {
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
        detail::generate_input_data_reduce<T>(local_size * elements_per_item, global_size);
    std::vector<T> plus_out_data(global_size);
    std::vector<T> mult_out_data(global_size);

    {
      sycl::buffer<T, 1> in_buf{in_data.data(), in_data.size()};
      sycl::buffer<T, 1> plus_out_buf{plus_out_data.data(), plus_out_data.size()};
      sycl::buffer<T, 1> mult_out_buf{mult_out_data.data(), mult_out_data.size()};

      sycl::queue queue;
      queue.submit([&](sycl::handler &cgh) {
        using namespace sycl::access;
        sycl::accessor in_acc{in_buf, cgh, sycl::read_only};
        sycl::accessor plus_out_acc{plus_out_buf, cgh, sycl::write_only};
        sycl::accessor mult_out_acc{mult_out_buf, cgh, sycl::write_only};

        cgh.parallel<detail::test_kernel<__LINE__, T>>(
            sycl::range<1>{num_groups}, sycl::range<1>{local_size}, [=](auto g) {
              auto group_id = g.get_group_linear_id();

              auto start = in_acc.get_pointer().get() + group_id * local_size * elements_per_item;
              auto end = start + local_size * elements_per_item;

              sycl::distribute_groups(g, [&](auto sg) {
                T plus_result = sycl::joint_reduce(sg, start, end, std::plus<T>());
                T mult_result = sycl::joint_reduce(sg, start, end, std::multiplies<T>());

                // store values into global memory
                sycl::distribute_items(sg, [&](sycl::s_item<1> idx) {
                  plus_out_acc[idx.get_global_linear_id()] = plus_result;
                  mult_out_acc[idx.get_global_linear_id()] = mult_result;
                });
              });
            });
      });
    }
    detail::check_output_joint_data_reduce(
        in_data, plus_out_data, local_size, num_groups, elements_per_item, 0);
    detail::check_output_joint_data_reduce(
        in_data, mult_out_data, local_size, num_groups, elements_per_item, 1);
  }
}

BOOST_AUTO_TEST_SUITE_END()

#endif
