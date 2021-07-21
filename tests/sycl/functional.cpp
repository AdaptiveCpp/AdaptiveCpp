/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2018, 2019 Aksel Alpay and contributors
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

#include "sycl_test_suite.hpp"

namespace s = cl::sycl;

BOOST_FIXTURE_TEST_SUITE(functional_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using known_identity_types = boost::mpl::list<float, double, char, signed char, unsigned char,
    int, long, unsigned int, unsigned long, bool>;

BOOST_AUTO_TEST_CASE_TEMPLATE(known_identities, T, known_identity_types::type) {
  static_assert(s::has_known_identity_v<s::minimum<T>, T>);
  static_assert(s::has_known_identity_v<s::minimum<>, T>);
  static_assert(s::has_known_identity_v<s::maximum<T>, T>);
  static_assert(s::has_known_identity_v<s::maximum<>, T>);

  BOOST_TEST((s::known_identity_v<s::plus<T>, T>) == T{0});
  BOOST_TEST((s::known_identity_v<s::plus<>, T>) == T{0});
  BOOST_TEST((s::known_identity_v<s::multiplies<T>, T>) == T{1});
  BOOST_TEST((s::known_identity_v<s::multiplies<>, T>) == T{1});

  if constexpr (std::is_floating_point_v<T>) {
    BOOST_TEST((s::known_identity_v<s::minimum<T>, T>) == std::numeric_limits<T>::infinity());
    BOOST_TEST((s::known_identity_v<s::minimum<>, T>) == std::numeric_limits<T>::infinity());
    BOOST_TEST((s::known_identity_v<s::maximum<T>, T>) == -std::numeric_limits<T>::infinity());
    BOOST_TEST((s::known_identity_v<s::maximum<>, T>) == -std::numeric_limits<T>::infinity());
  } else {
    BOOST_TEST((s::known_identity_v<s::minimum<T>, T>) == std::numeric_limits<T>::max());
    BOOST_TEST((s::known_identity_v<s::minimum<>, T>) == std::numeric_limits<T>::max());
    BOOST_TEST((s::known_identity_v<s::maximum<T>, T>) == std::numeric_limits<T>::lowest());
    BOOST_TEST((s::known_identity_v<s::maximum<>, T>) == std::numeric_limits<T>::lowest());
  }

  if constexpr (std::is_integral_v<T>) {
    static_assert(s::has_known_identity_v<s::bit_or<T>, T>);
    static_assert(s::has_known_identity_v<s::bit_or<>, T>);
    static_assert(s::has_known_identity_v<s::bit_xor<T>, T>);
    static_assert(s::has_known_identity_v<s::bit_xor<>, T>);

    BOOST_TEST((s::known_identity_v<s::bit_or<T>, T>) == T{0});
    BOOST_TEST((s::known_identity_v<s::bit_or<>, T>) == T{0});
    BOOST_TEST((s::known_identity_v<s::bit_xor<T>, T>) == T{0});
    BOOST_TEST((s::known_identity_v<s::bit_xor<>, T>) == T{0});
  }

  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    static_assert(s::has_known_identity_v<s::bit_and<T>, T>);
    static_assert(s::has_known_identity_v<s::bit_and<>, T>);

    BOOST_TEST((s::known_identity_v<s::bit_and<T>, T>) == static_cast<T>(~T{0}));
    BOOST_TEST((s::known_identity_v<s::bit_and<>, T>) == static_cast<T>(~T{0}));
  }

  if constexpr (std::is_same_v<T, bool>) {
    static_assert(s::has_known_identity_v<s::bit_and<T>, T>);
    static_assert(s::has_known_identity_v<s::bit_and<>, T>);
    static_assert(s::has_known_identity_v<s::logical_or<T>, T>);
    static_assert(s::has_known_identity_v<s::logical_or<>, T>);
    static_assert(s::has_known_identity_v<s::logical_and<T>, T>);
    static_assert(s::has_known_identity_v<s::logical_and<>, T>);

    BOOST_TEST((s::known_identity_v<s::bit_and<T>, T>) == true);
    BOOST_TEST((s::known_identity_v<s::bit_and<>, T>) == true);
    BOOST_TEST((s::known_identity_v<s::logical_or<T>, T>) == false);
    BOOST_TEST((s::known_identity_v<s::logical_or<>, T>) == false);
    BOOST_TEST((s::known_identity_v<s::logical_and<T>, T>) == true);
    BOOST_TEST((s::known_identity_v<s::logical_and<>, T>) == true);
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
