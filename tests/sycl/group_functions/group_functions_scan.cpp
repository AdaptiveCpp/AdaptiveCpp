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

BOOST_FIXTURE_TEST_SUITE(group_functions_tests, reset_device_fixture)

BOOST_AUTO_TEST_CASE_TEMPLATE(group_exclusive_scan, T, test_types) {
  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size;
  const size_t offset_divisor = global_size;
  const size_t buffer_size    = global_size;
  const auto data_generator   = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_exclusive_scan(g, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = T{};
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);

          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_exclusive_scan(g, local_value, detail::initialize_type<T>(10),
                                     std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  if constexpr (!std::is_floating_point<T>::value) {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_exclusive_scan(g, local_value, detail::initialize_type<T>(10),
                                     std::multiplies<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] * vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init multiplication in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_exclusive_scan_ptr, T, test_types) {
  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t buffer_size    = global_size * 4;
  const size_t offset_margin  = global_size * 2;
  const size_t offset_divisor = global_size * 2;
  const auto data_generator   = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::detail::exclusive_scan(g, start.get(), end.get(), out.get(), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = T{};
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j - 1];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + global_size * 2];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);

          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::detail::exclusive_scan(g, start.get(), end.get(), out.get(),
                                   detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j - 1];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + global_size * 2];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HIP)
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_exclusive_scan, T, test_types) {
  const uint32_t subgroup_size = static_cast<uint32_t>(warpSize);
  const size_t local_size      = subgroup_size;
  const size_t global_size     = subgroup_size * 4;
  const size_t offset_margin   = global_size;
  const size_t offset_divisor  = global_size;
  const size_t buffer_size     = global_size;
  const auto data_generator    = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_exclusive_scan(sg, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = T{};
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);

          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_exclusive_scan(sg, local_value, detail::initialize_type<T>(10),
                                     std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j - 1];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }
}
#endif

BOOST_AUTO_TEST_CASE_TEMPLATE(group_inclusive_scan, T, test_types) {
  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t offset_margin  = global_size;
  const size_t offset_divisor = global_size;
  const size_t buffer_size    = global_size;
  const auto data_generator   = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_inclusive_scan(g, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  if constexpr (!std::is_floating_point<T>::value) {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_inclusive_scan(g, local_value, std::multiplies<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] * vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: no init multiplication in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_inclusive_scan(g, local_value, detail::initialize_type<T>(10),
                                     std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size] + detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(group_inclusive_scan_ptr, T, test_types) {
  const size_t local_size     = 256;
  const size_t global_size    = 1024;
  const size_t local_size_x   = 16;
  const size_t local_size_y   = 16;
  const size_t global_size_x  = 32;
  const size_t global_size_y  = 32;
  const size_t buffer_size    = global_size * 4;
  const size_t offset_margin  = global_size * 2;
  const size_t offset_divisor = global_size * 2;
  const auto data_generator   = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::detail::inclusive_scan(g, start.get(), end.get(), out.get(), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] = vOrig[i * 2 * local_size];
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + 2 * global_size];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      auto local_size = g.get_local_range().size();
      auto start = acc.get_pointer() + (global_linear_id / local_size) * local_size * 2;
      auto end   = start + local_size * 2;
      auto out   = acc.get_pointer() + 2 * 4 * local_size +
                 (global_linear_id / local_size) * local_size * 2;

      sycl::detail::inclusive_scan(g, start.get(), end.get(), out.get(),
                                   detail::initialize_type<T>(10), std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * 2 * local_size] =
            vOrig[i * 2 * local_size] + detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size * 2; ++j)
          expected[i * 2 * local_size + j] =
              expected[i * 2 * local_size + j - 1] + vOrig[i * 2 * local_size + j];

        for (size_t j = i * 2 * local_size; j < (i + 1) * local_size * 2; ++j) {
          T computed = vIn[j + 2 * global_size];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);

    test_nd_group_function_2d<__LINE__, T>(local_size_x, local_size_y, global_size_x,
                                           global_size_y, offset_margin, offset_divisor,
                                           buffer_size, data_generator, tested_function,
                                           validation_function);
  }
}

#if defined(HIPSYCL_PLATFORM_CUDA) || defined(HIPSYCL_PLATFORM_HIP)
BOOST_AUTO_TEST_CASE_TEMPLATE(sub_group_inclusive_scan, T, test_types) {
  const uint32_t subgroup_size = static_cast<uint32_t>(warpSize);
  const size_t local_size      = subgroup_size;
  const size_t global_size     = subgroup_size * 4;
  const size_t offset_margin   = global_size;
  const size_t offset_divisor  = global_size;
  const size_t buffer_size     = global_size;
  const auto data_generator    = [](std::vector<T> &v) {
    for (size_t i = 0; i < global_size; ++i)
      v[i] = detail::initialize_type<T>(i) +
             detail::get_offset<T>(offset_margin, offset_divisor);
  };

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] = sycl::group_inclusive_scan(sg, local_value, std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size];
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }

  {
    const auto tested_function = [](auto acc, size_t global_linear_id, sycl::sub_group sg,
                                    auto g, T local_value) {
      acc[global_linear_id] =
          sycl::group_inclusive_scan(sg, local_value, detail::initialize_type<T>(10),
                                     std::plus<T>());
    };
    const auto validation_function = [](const std::vector<T> &vIn,
                                        const std::vector<T> &vOrig) {
      std::vector<T> expected(vOrig.size());

      for (size_t i = 0; i < global_size / local_size; ++i) {
        expected[i * local_size] = vOrig[i * local_size] + detail::initialize_type<T>(10);
        for (size_t j = 1; j < local_size; ++j)
          expected[i * local_size + j] =
              expected[i * local_size + j - 1] + vOrig[i * local_size + j];

        for (size_t j = i * local_size; j < (i + 1) * local_size; ++j) {
          T computed = vIn[j];
          BOOST_TEST(detail::compare_type(expected[j], computed),
                     detail::type_to_string(computed)
                         << " at position " << j << " instead of "
                         << detail::type_to_string(expected[j])
                         << " for case: init in group " << i);
          if (!detail::compare_type(expected[j], computed))
            break;
        }
      }
    };

    test_nd_group_function_1d<__LINE__, T>(local_size, global_size, offset_margin,
                                           offset_divisor, buffer_size, data_generator,
                                           tested_function, validation_function);
  }
}
#endif
BOOST_AUTO_TEST_SUITE_END()
