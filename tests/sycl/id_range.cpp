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

#include <cstdint>

#include "sycl_test_suite.hpp"

BOOST_FIXTURE_TEST_SUITE(id_range_tests, reset_device_fixture)

template <template<int D> class T, int dimensions>
void test_id_range_operators() {
  const auto test_value = make_test_value<T, dimensions>({ 5 }, { 5, 7 }, { 5, 7, 11 });
  const auto other_test_value = make_test_value<T, dimensions>({ 3 }, { 3, 4 }, { 3, 4, 9 });

  {
    // T + T
    const auto result = test_value + other_test_value;
    if(dimensions >= 1) BOOST_TEST(result[0] == 8);
    if(dimensions >= 2) BOOST_TEST(result[1] == 11);
    if(dimensions == 3) BOOST_TEST(result[2] == 20);
  }

  {
    // T + size_t
    const auto result = test_value + 2;
    if(dimensions >= 1) BOOST_TEST(result[0] == 7);
    if(dimensions >= 2) BOOST_TEST(result[1] == 9);
    if(dimensions == 3) BOOST_TEST(result[2] == 13);
  }

  {
    // T += T
    auto result = test_value;
    result+= other_test_value;
    if(dimensions >= 1) BOOST_TEST(result[0] == 8);
    if(dimensions >= 2) BOOST_TEST(result[1] == 11);
    if(dimensions == 3) BOOST_TEST(result[2] == 20);
  }

  {
    // T += size_t
    auto result = test_value;
    result += 2;
    if(dimensions >= 1) BOOST_TEST(result[0] == 7);
    if(dimensions >= 2) BOOST_TEST(result[1] == 9);
    if(dimensions == 3) BOOST_TEST(result[2] == 13);
  }

  {
    // size_t + T
    auto result = 2 + test_value;
    if(dimensions >= 1) BOOST_TEST(result[0] == 7);
    if(dimensions >= 2) BOOST_TEST(result[1] == 9);
    if(dimensions == 3) BOOST_TEST(result[2] == 13);
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(range_api, _dimensions, test_dimensions::type) {
  namespace s = sycl;
  constexpr auto d = _dimensions::value;

  const auto test_value = make_test_value<s::range, d>({ 5 }, { 5, 7 }, { 5, 7, 11 });

  // --- Common by-value semantics ---

  {
    // Copy constructor
    s::range<d> range(test_value);
    assert_array_equality(range, test_value);
  }
  {
    // Move constructor
    s::range<d> range(([&]() {
      s::range<d> copy(test_value);
      return std::move(copy);
    })());
    assert_array_equality(range, test_value);
  }
  {
    // Copy assignment
    s::range<d> range;
    range = test_value;
    assert_array_equality(range, test_value);
  }
  {
    // Move assignment
    s::range<d> range;
    range = ([&]() {
      s::range<d> copy(test_value);
      return std::move(copy);
    })();
    assert_array_equality(range, test_value);
  }
  {
    // Equality operator
    s::range<d> range;
    BOOST_TEST(!(range == test_value));
    range = test_value;
    BOOST_TEST((range == test_value));
  }
  {
    // Inequality operator
    s::range<d> range;
    BOOST_TEST((range != test_value));
    range = test_value;
    BOOST_TEST(!(range != test_value));
  }

  // --- range-specific API ---

  {
    // range::get()
    const auto range = test_value;
    if(d >= 1) BOOST_TEST(range.get(0) == 5);
    if(d >= 2) BOOST_TEST(range.get(1) == 7);
    if(d == 3) BOOST_TEST(range.get(2) == 11);
  }
  {
    // range::operator[]
    auto range = test_value;
    if(d >= 1) range[0] += 2;
    if(d >= 2) range[1] += 3;
    if(d == 3) range[2] += 5;
    assert_array_equality(range, make_test_value<s::range, d>(
      { 7 }, { 7, 10 }, { 7, 10, 16 }));
  }
  {
    // const range::operator[]
    const auto range = test_value;
    if(d >= 1) BOOST_TEST(range[0] == 5);
    if(d >= 2) BOOST_TEST(range[1] == 7);
    if(d == 3) BOOST_TEST(range[2] == 11);
  }
  {
    // range::size()
    const auto range = test_value;
    BOOST_TEST(range.size() == 5 * (d >= 2 ? 7 : 1) * (d == 3 ? 11 : 1));
  }

  test_id_range_operators<s::range, d>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(id_api, _dimensions, test_dimensions::type) {
  namespace s = sycl;
  constexpr auto d = _dimensions::value;

  const auto test_value = make_test_value<s::id, d>(
    { 5 }, { 5, 7 }, { 5, 7, 11 });

  // --- Common by-value semantics ---

  {
    // Copy constructor
    s::id<d> id(test_value);
    assert_array_equality(id, test_value);
  }
  {
    // Move constructor
    s::id<d> id(([&]() {
      s::id<d> copy(test_value);
      return std::move(copy);
    })());
    assert_array_equality(id, test_value);
  }
  {
    // Copy assignment
    s::id<d> id;
    id = test_value;
    assert_array_equality(id, test_value);
  }
  {
    // Move assignment
    s::id<d> id;
    id = ([&]() {
      s::id<d> copy(test_value);
      return std::move(copy);
    })();
    assert_array_equality(id, test_value);
  }
  {
    // Equality operator
    s::id<d> id;
    BOOST_TEST(!(id == test_value));
    id = test_value;
    BOOST_TEST((id == test_value));
  }
  {
    // Inequality operator
    s::id<d> id;
    BOOST_TEST((id != test_value));
    id = test_value;
    BOOST_TEST(!(id != test_value));
  }

  // --- id-specific API ---

  {
    const auto test_range = make_test_value<s::range, d>(
      { 5 }, { 5, 7 }, { 5, 7, 11 });
    s::id<d> id{test_range};
    assert_array_equality(id, test_value);
  }
  {
    // TODO: Test conversion from item
    // (This is a bit annoying as items can only be constructed on a __device__)
  }
  {
    // id::get()
    const auto id = test_value;
    if(d >= 1) BOOST_TEST(id.get(0) == 5);
    if(d >= 2) BOOST_TEST(id.get(1) == 7);
    if(d == 3) BOOST_TEST(id.get(2) == 11);
  }
  {
    // id::operator[]
    auto id = test_value;
    if(d >= 1) id[0] += 2;
    if(d >= 2) id[1] += 3;
    if(d == 3) id[2] += 5;
    assert_array_equality(id, make_test_value<s::id, d>(
      { 7 }, { 7, 10 }, { 7, 10, 16 }));
  }
  {
    // const id::operator[]
    const auto id = test_value;
    if(d >= 1) BOOST_TEST(id[0] == 5);
    if(d >= 2) BOOST_TEST(id[1] == 7);
    if(d == 3) BOOST_TEST(id[2] == 11);
  }

  test_id_range_operators<s::id, d>();
}


BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
