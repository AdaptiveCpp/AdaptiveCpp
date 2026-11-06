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

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#include <sycl/sycl.hpp>
#include <numeric>

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_mul, T, test_types) {
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }
  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = (i % local_size == 0) ? T{static_cast<T>(2)} : T{static_cast<T>(1)};
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, std::multiplies<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = expected * vOrig[i * local_size + j];

        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected) << " for group " << i
                         << " for local_size " << local_size
                         << " and case: no init multiplication");
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce, T, test_types) {
  SKIP_IF_MOLTENVK(sycl::device{})
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }

  const size_t elements_per_thread = 1;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] =
          detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = T{};
        for (size_t j = 0; j < local_size; ++j)
          expected = expected + vOrig[i * local_size + j];

        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected) << " for group " << i
                         << " for local_size " << local_size << " and case: no init");
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(
          g, local_value, detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = detail::initialize_type<T>(10);
        for (size_t j = 0; j < local_size; ++j)
          expected = expected + vOrig[i * local_size + j];

        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected) << " for group " << i
                         << " for local_size " << local_size << " and case: init");
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_ptr, T, test_types) {
  SKIP_IF_MOLTENVK(sycl::device{})
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }

  const size_t elements_per_thread = 3;
  const auto   data_generator      = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(global_size * elements_per_thread, local_size * 2);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size  = g.get_local_range().size();
      auto global_size = local_size * 4;
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      T local = sycl::joint_reduce(g, start.get(), end.get(), std::plus<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = T{};
        for (size_t j = 0; j < local_size * 2; ++j)
          expected = expected + vOrig[i * 2 * local_size + j];

        T computed = vIn[2 * global_size + i * local_size];
        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for local_size "
                       << local_size << " and case: no init");
        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size  = g.get_local_range().size();
      auto global_size = local_size * 4;
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;

      T local = sycl::joint_reduce(g, start.get(), end.get(),
                                     detail::initialize_type<T>(10), std::plus<T>());
      sycl::group_barrier(g);
      acc[global_linear_id + 2 * global_size] = local;
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = detail::initialize_type<T>(10);
        for (size_t j = 0; j < local_size * 2; ++j)
          expected = expected + vOrig[i * 2 * local_size + j];

        T computed = vIn[i * local_size + 2 * global_size];
        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for local_size "
                       << local_size << " and case: init");
        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_reduce, T, test_types) {
  sycl::device dev;
  SKIP_IF_MOLTENVK(dev)

  if (!dev.is_host()) {
    const size_t   elements_per_thread = 1;
    const auto     data_generator      = [](std::vector<T> &v, size_t local_size,
                                  size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] =
            detail::initialize_type<T>(i) + detail::get_offset<T>(global_size, global_size);
    };

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::reduce_over_group(sg, local_value, sycl::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig,size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < global_size / local_size; ++i) {
          T    expected         = T{};
          auto actual_warp_size = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 0; j < actual_warp_size; ++j)
            expected = expected + vOrig[i * local_size + j];

          T computed = vIn[i * local_size];
          BOOST_TEST(detail::compare_type(expected, computed),
                    detail::type_to_string(computed)
                        << " at position " << i << " instead of "
                        << detail::type_to_string(expected) << " for local_size "
                        << local_size << " and case: no init");
          if (!detail::compare_type(expected, computed))
            break;
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }

    {
      const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                      auto g, T local_value) {
        acc[global_linear_id] = sycl::reduce_over_group(
            sg, local_value, detail::initialize_type<T>(10), sycl::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t subgroup_size, size_t local_size,
                                          size_t global_size) {
        for (size_t i = 0; i < global_size / local_size; ++i) {
          T    expected         = detail::initialize_type<T>(10);
          auto actual_warp_size = local_size < subgroup_size ? local_size : subgroup_size;
          for (size_t j = 0; j < actual_warp_size; ++j)
            expected = expected + vOrig[i * local_size + j];

          T computed = vIn[i * local_size];
          BOOST_TEST(detail::compare_type(expected, computed),
                    detail::type_to_string(computed)
                        << " at position " << i << " instead of "
                        << detail::type_to_string(expected) << " for local_size "
                        << local_size << " and case: init");
          if (!detail::compare_type(expected, computed))
            break;
        }
      };

      test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_min, T, test_types) {
  const size_t elements_per_thread = 1;
  if constexpr (std::is_arithmetic_v<T>) {
    const auto data_generator = [](std::vector<T> &v, size_t local_size, size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = detail::initialize_type<T>(i % local_size + 1);
    };
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, sycl::minimum<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = sycl::minimum<T>{}(expected, vOrig[i * local_size + j]);
        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                      detail::type_to_string(computed) << " at position " << j
                      << " instead of " << detail::type_to_string(expected)
                      << " for group " << i << " local_size " << local_size);
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };
    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
  } else {
    BOOST_TEST_MESSAGE("Skipping group_reduce_min for non-scalar type");
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_max, T, test_types) {
  const size_t elements_per_thread = 1;
  if constexpr (std::is_arithmetic_v<T>) {
    const auto data_generator = [](std::vector<T> &v, size_t local_size, size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = detail::initialize_type<T>(i % local_size + 1);
    };
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, sycl::maximum<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = sycl::maximum<T>{}(expected, vOrig[i * local_size + j]);
        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                      detail::type_to_string(computed) << " at position " << j
                      << " instead of " << detail::type_to_string(expected)
                      << " for group " << i << " local_size " << local_size);
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };
    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                            tested_function, validation_function);
  } else {
    BOOST_TEST_MESSAGE("Skipping group_reduce_max for non-scalar type");
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_bit_and, T, test_types) {
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }

  const size_t elements_per_thread = 1;
  if constexpr(std::is_integral_v<detail::elementType<T>>) {
    const auto data_generator = [](std::vector<T> &v, size_t local_size, size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i) {
        size_t local_idx = i % local_size;
        v[i] = (local_idx == 0) ? detail::initialize_type<T>(0x0F) : ~T{};
      }
    };
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, std::bit_and<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = expected & vOrig[i * local_size + j];
        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed) << " at position " << j
                     << " instead of " << detail::type_to_string(expected)
                     << " for group " << i << " local_size " << local_size);
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };
    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  } else {
    BOOST_TEST_MESSAGE("Skipping group_reduce_bit_and for non-integer type");
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_bit_or, T, test_types) {
  SKIP_IF_MOLTENVK(sycl::device{})
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }

  const size_t elements_per_thread = 1;
  if constexpr(std::is_integral_v<detail::elementType<T>>) {
    const auto data_generator = [](std::vector<T> &v, size_t local_size, size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i) {
        size_t local_idx = i % local_size;
        v[i] = (local_idx == 0) ? detail::initialize_type<T>(0x5A) : T{};
      }
    };
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, std::bit_or<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = expected | vOrig[i * local_size + j];
        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed) << " at position " << j
                     << " instead of " << detail::type_to_string(expected)
                     << " for group " << i << " local_size " << local_size);
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };
    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  } else {
    BOOST_TEST_MESSAGE("Skipping group_reduce_bit_or for non-integer type");
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_bit_xor, T, test_types) {
  SKIP_IF_MOLTENVK(sycl::device{})
  if (sycl::device{}.get_backend() == sycl::backend::vk &&
      !std::is_scalar_v<T>) {
    BOOST_TEST_MESSAGE("group functions not yet stable on VK backend");
    return;
  }

  const size_t elements_per_thread = 1;
  if constexpr(std::is_integral_v<detail::elementType<T>>) {
    const auto data_generator = [](std::vector<T> &v, size_t local_size, size_t global_size) {
      for (size_t i = 0; i < v.size(); ++i)
        v[i] = detail::initialize_type<T>(i % local_size + 1);
    };
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::reduce_over_group(g, local_value, std::bit_xor<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig, size_t, size_t local_size,
                                        size_t global_size) {
      for (size_t i = 0; i < global_size / local_size; ++i) {
        T expected = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected = expected ^ vOrig[i * local_size + j];
        for (size_t j = 0; j < local_size; ++j) {
          T computed = vIn[i * local_size + j];
          BOOST_TEST(detail::compare_type(expected, computed),
                     detail::type_to_string(computed) << " at position " << j
                     << " instead of " << detail::type_to_string(expected)
                     << " for group " << i << " local_size " << local_size);
          if (!detail::compare_type(expected, computed))
            break;
        }
      }
    };
    test_nd_group_function_1d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
    test_nd_group_function_2d<__LINE__, T>(elements_per_thread, data_generator,
                                           tested_function, validation_function);
  } else {
    BOOST_TEST_MESSAGE("Skipping group_reduce_bit_xor for non-integer type");
  }
}

template <typename T, typename BinaryOperation>
void test_joint_reduce_min_max() {
  SKIP_IF_MOLTENVK(sycl::device{})
  if constexpr (std::is_arithmetic_v<T>) {
    sycl::queue q;
    if constexpr (std::is_same_v<T, double>) {
      if (!q.get_device().has(sycl::aspect::fp64))
        return;
    }
    constexpr size_t group_size = 128;
    constexpr size_t N          = 20; 

    std::vector<T> host_in(N);
    T *dev_in  = sycl::malloc_device<T>(N, q);
    T *dev_out = sycl::malloc_device<T>(1, q);

    auto run_and_check = [&](T expected) {
      T host_out{};
      q.copy(host_in.data(), dev_in, N).wait();
      q.parallel_for(sycl::nd_range<1>{{group_size}, {group_size}},
                     [=](sycl::nd_item<1> it) {
                       *dev_out = sycl::joint_reduce(
                           it.get_group(), dev_in, dev_in + N, BinaryOperation{});
                     })
          .wait_and_throw();
      q.copy(dev_out, &host_out, 1).wait();
      BOOST_TEST(host_out == expected);
    };

    for (size_t i = 0; i < N; ++i) {
      // Do not include 0 in the test data
      T val      = i + 1;
      host_in[i] = std::is_signed_v<T> ? -val : val;
    }
    run_and_check(std::accumulate(host_in.begin() + 1, host_in.end(),
                                  host_in[0], BinaryOperation{}));

    if constexpr (std::is_floating_point_v<T>) {
      constexpr T inf = std::is_same_v<BinaryOperation, sycl::minimum<T>>
                            ? std::numeric_limits<T>::infinity()
                            : -std::numeric_limits<T>::infinity();
      std::fill(host_in.begin(), host_in.end(), inf);
      run_and_check(inf);
    }

    sycl::free(dev_in, q);
    sycl::free(dev_out, q);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_ptr_min, T, test_types) {
  test_joint_reduce_min_max<T, sycl::minimum<T>>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_ptr_max, T, test_types) {
  test_joint_reduce_min_max<T, sycl::maximum<T>>();
}

BOOST_AUTO_TEST_SUITE_END()

#endif

