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

#include "../sycl_test_suite.hpp"
#include "group_functions.hpp"

#ifdef HIPSYCL_ENABLE_GROUP_ALGORITHM_TESTS

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(group_reduce_mul, T, test_types) {
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
                                        const std::vector<T> &vOrig, size_t local_size,
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
                                        const std::vector<T> &vOrig, size_t local_size,
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
                                        const std::vector<T> &vOrig, size_t local_size,
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
                                        const std::vector<T> &vOrig, size_t local_size,
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
                                        const std::vector<T> &vOrig, size_t local_size,
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
  if(!sycl::queue{}.get_device().is_host()) {
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
        acc[global_linear_id] = sycl::reduce_over_group(sg, local_value, std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t local_size,
                                          size_t global_size) {
        auto subgroup_size = detail::get_subgroup_size();
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
            sg, local_value, detail::initialize_type<T>(10), std::plus<T>());
      };
      const auto validation_function = [](const std::vector<T> &vIn,
                                          const std::vector<T> &vOrig, size_t local_size,
                                          size_t global_size) {
        auto subgroup_size = detail::get_subgroup_size();
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

BOOST_AUTO_TEST_SUITE_END()

#endif
