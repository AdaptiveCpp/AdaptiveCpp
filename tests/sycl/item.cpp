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


#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(item_tests, reset_device_fixture)


BOOST_AUTO_TEST_CASE_TEMPLATE(item_api, _dimensions, test_dimensions::type) {
  namespace s = cl::sycl;
  // Specify type explicitly to workaround Clang bug #45538
  constexpr int d = _dimensions::value;

  const auto test_range = make_test_value<s::range, d>(
    { 5 }, { 5, 7 }, { 5, 7, 11 });

  // TODO: Add tests for common by-value semantics

  s::queue queue;

  {
    // item::get_id and item::operator[] without offset

    s::buffer<s::id<d>, d> result1{test_range};
    s::buffer<s::id<d>, d> result2{test_range};
    s::buffer<s::id<d>, d> result3{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      auto acc3 = result3.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_id, d>>(test_range,
        [=](const s::item<d> item) {
          {
            acc1[item] = item.get_id();
          }
          {
            s::id<d> id2;
            if(d >= 1) id2[0] = item.get_id(0);
            if(d >= 2) id2[1] = item.get_id(1);
            if(d == 3) id2[2] = item.get_id(2);
            acc2[item] = id2;
          }
          {
            s::id<d> id3;
            if(d >= 1) id3[0] = item[0];
            if(d >= 2) id3[1] = item[1];
            if(d == 3) id3[2] = item[2];
            acc3[item] = id3;
          }
        });
      });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    auto acc3 = result3.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], id);
          assert_array_equality(acc2[id], id);
          assert_array_equality(acc3[id], id);
        }
      }
    }
  }
  {
    // item::get_id and item::operator[] with offset

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<s::id<d>, d> result1{test_range + test_offset};
    s::buffer<s::id<d>, d> result2{test_range + test_offset};
    s::buffer<s::id<d>, d> result3{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      auto acc3 = result3.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_id_offset, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          {
            acc1[item] = item.get_id();
          }
          {
            s::id<d> id2;
            if(d >= 1) id2[0] = item.get_id(0);
            if(d >= 2) id2[1] = item.get_id(1);
            if(d == 3) id2[2] = item.get_id(2);
            acc2[item] = id2;
          }
          {
            s::id<d> id3;
            if(d >= 1) id3[0] = item[0];
            if(d >= 2) id3[1] = item[1];
            if(d == 3) id3[2] = item[2];
            acc3[item] = id3;
          }
        });
    });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    auto acc3 = result3.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], id);
          assert_array_equality(acc2[id], id);
          assert_array_equality(acc3[id], id);
        }
      }
    }
  }
  {
    // item::get_range

    s::buffer<s::range<d>, d> result1{test_range};
    s::buffer<s::range<d>, d> result2{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc1 = result1.template get_access<s::access::mode::discard_write>(cgh);
      auto acc2 = result2.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_range, d>>(test_range,
        [=](const s::item<d> item) {
          {
            acc1[item] = item.get_range();
          }
          {
            auto range2 = item.get_range();
            if(d >= 1) range2[0] = item.get_range(0);
            if(d >= 2) range2[1] = item.get_range(1);
            if(d == 3) range2[2] = item.get_range(2);
            acc2[item] = range2;
          }
        });
      });

    auto acc1 = result1.template get_access<s::access::mode::read>();
    auto acc2 = result2.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc1[id], test_range);
          assert_array_equality(acc2[id], test_range);
        }
      }
    }
  }
  {
    // item::get_offset

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<s::id<d>, d> result{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_offset, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          acc[item] = item.get_offset();
        });
    });

    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc[id], s::id<d>(test_offset));
        }
      }
    }
  }
  {
    // Conversion operator from item<d, false> to item<d, true>

    s::buffer<s::id<d>, d> result{test_range};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_conversion, d>>(test_range,
        [=](const s::item<d> item) {
          acc[item] = item.get_offset();
        });
    });

    const auto empty_offset = make_test_value<s::id, d>({0}, {0, 0}, {0, 0, 0});
    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = 0; i < test_range[0]; ++i) {
      for(size_t j = 0; j < (d >= 2 ? test_range[1] : 1); ++j) {
        for(size_t k = 0; k < (d == 3 ? test_range[2] : 1); ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          assert_array_equality(acc[id], empty_offset);
        }
      }
    }
  }
  {
    // item::get_linear_id

    // Make offset a range as it's easier to handle (and can be converted to id)
    const auto test_offset = make_test_value<s::range, d>(
      { 2 }, { 2, 3 }, { 2, 3, 5 });

    s::buffer<size_t, d> result{test_range + test_offset};

    queue.submit([&](s::handler& cgh) {
      auto acc = result.template get_access<s::access::mode::discard_write>(cgh);
      cgh.parallel_for<kernel_name<class item_get_linear_id, d>>(test_range,
        s::id<d>(test_offset), [=](const s::item<d> item) {
          acc[item] = item.get_linear_id();
        });
    });

    auto acc = result.template get_access<s::access::mode::read>();
    for(size_t i = test_offset[0]; i < test_range[0]; ++i) {
      const auto ja = d >= 2 ? test_offset[1] : 0;
      const auto jb = d >= 2 ? test_range[1] + ja : 1;
      for(size_t j = ja; j < jb; ++j) {
        const auto ka = d == 3 ? test_offset[2] : 0;
        const auto kb = d == 3 ? test_range[2] + ka : 1;
        for(size_t k = ka; k < kb; ++k) {
          const auto id = make_test_value<s::id, d>({i}, {i, j}, {i, j, k});
          const size_t linear_id = i * (jb - ja) * (kb - ka) + j * (kb - ka) + k;
          BOOST_REQUIRE(acc[id] == linear_id);
        }
      }
    }
  }
}


BOOST_AUTO_TEST_SUITE_END()
