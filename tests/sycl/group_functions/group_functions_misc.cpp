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
 * this list of conditions and the following disclaimer.
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

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE(group_barrier) {
  using T = int;

  const size_t elements_per_thread = 1;
  const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i);
  };

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g,
                                     T local_value) {
      int tmp = -10000;
      size_t local_id = g.get_local_linear_id();
      auto local_size = g.get_local_range().size();
      size_t group_offset = (global_linear_id / local_size) * local_size;
      for (int i = 0; i < local_size; ++i) {
        if (local_id == i) {
          for (int j = 0; j < 10000; ++j)
            tmp++;
        }
        sycl::group_barrier(g);
        if (local_id == i)
          acc[group_offset + i] = tmp;
        sycl::group_barrier(g);
        tmp = acc[group_offset + i];
      }
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = (i % local_size) * 10000;
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for group: " << i / local_size);

        //        if (!detail::compare_type(expected, computed))
        //          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_broadcast, T, test_types) {
  const size_t elements_per_thread = 1;
  const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
  };

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g,
                                     T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected =
            detail::initialize_type<T>(((int)i / local_size) * local_size) +
            detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: no id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }

  {
    const auto tested_function = [=](auto acc, size_t global_linear_id,
                                     sycl::sub_group sg, auto g,
                                     T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(g, local_value, 10);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = detail::initialize_type<T>(
                         ((int)i / local_size) * local_size + 10) +
                     detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: linear id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }

  {
    const auto tested_function_1d = [](auto acc, size_t global_linear_id,
                                       sycl::sub_group sg, auto g,
                                       T local_value) {
      acc[global_linear_id] =
          sycl::group_broadcast(g, local_value, sycl::id<1>(10));
    };
    const auto tested_function_2d = [](auto acc, size_t global_linear_id,
                                       sycl::sub_group sg, auto g,
                                       T local_value) {
      acc[global_linear_id] =
          sycl::group_broadcast(g, local_value, sycl::id<2>(0, 10));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        T expected = detail::initialize_type<T>(
                         ((int)i / local_size) * local_size + 10) +
                     detail::get_offset<T>(global_size, 1);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function_1d, validation_function);

    test_nd_group_function_2d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function_2d, validation_function);
  }
}

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HIP)
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_broadcast, T, test_types) {
  const size_t elements_per_thread = 1;
  const uint32_t subgroup_size = static_cast<uint32_t>(warpSize);

  const auto data_generator = [](std::vector<T> &v, size_t local_size,
                                 size_t global_size) {
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = detail::initialize_type<T>(i) + detail::get_offset<T>(global_size);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id,
                                    sycl::sub_group sg, auto g, T local_value) {
      acc[global_linear_id] = sycl::group_broadcast(sg, local_value);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        int expected_base = i%local_size;
        expected_base = ((int)expected_base/warpSize) * warpSize;
        expected_base += ((int)i/local_size) * local_size;

        T expected = detail::initialize_type<T>(expected_base) + detail::get_offset<T>(global_size);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: no id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id,
                                    sycl::sub_group sg, auto g, T local_value) {
        acc[global_linear_id] = sycl::group_broadcast(sg, local_value, 10);
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        int expected_base = i%local_size;
        expected_base = ((int)expected_base/warpSize) * warpSize;
        expected_base += ((int)i/local_size) * local_size + 10;

        T expected = detail::initialize_type<T>(expected_base) + detail::get_offset<T>(global_size);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected)
                       << " for case: linear id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id,
                                    sycl::sub_group sg, auto g, T local_value) {
        acc[global_linear_id] =
            sycl::group_broadcast(sg, local_value, sycl::id<1>(10));
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig,
                                        size_t local_size, size_t global_size) {
      for (size_t i = 0; i < vIn.size(); ++i) {
        int expected_base = i%local_size;
        expected_base = ((int)expected_base/warpSize) * warpSize;
        expected_base += ((int)i/local_size) * local_size + 10;

        T expected = detail::initialize_type<T>(expected_base) + detail::get_offset<T>(global_size);
        T computed = vIn[i];

        BOOST_TEST(detail::compare_type(expected, computed),
                   detail::type_to_string(computed)
                       << " at position " << i << " instead of "
                       << detail::type_to_string(expected) << " for case: id");

        if (!detail::compare_type(expected, computed))
          break;
      }
    };

    test_nd_group_function_1d<__LINE__, T>(
        elements_per_thread, data_generator, tested_function, validation_function);
  }
}
#endif
BOOST_AUTO_TEST_SUITE_END()
