/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay
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

#include <algorithm>
#include <execution>
#include <utility>
#include <vector>
#include <atomic>

#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

#include "pstl_test_suite.hpp"


using atomic_int_test_types = boost::mpl::list<
  int, unsigned, long long, unsigned long long>;

using atomic_test_types = boost::mpl::list<
  int, unsigned, long long, unsigned long long, float, double>;

BOOST_FIXTURE_TEST_SUITE(pstl_std_atomic, enable_unified_shared_memory)

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_load_store, T, atomic_test_types::type) {
  std::vector<std::atomic<T>> data(1);
  data[0] = T{42};

  
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), [](auto& x){
    auto new_val = x.load(std::memory_order_relaxed) + T{1};
    x.store(new_val, std::memory_order_relaxed);
  });

  BOOST_CHECK(data[0] == T{43});
}


BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_exchange, T, atomic_test_types::type) {
  std::vector<std::atomic<T>> data(1);
  data[0] = T{42};

  
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), [](auto& x){
    auto old_val = x.exchange(1, std::memory_order_relaxed);
    auto current_val = x.load(std::memory_order_relaxed);
    x.store(old_val + current_val, std::memory_order_relaxed);
  });

  BOOST_CHECK(data[0] == T{43});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_cmp_exchange, T, atomic_test_types::type) {
  std::vector<std::atomic<T>> data(1);
  data[0] = T{42};
  
  std::vector<int> successes(2);
  int* success_ptr = successes.data();

  std::vector<T> expected_data(2);
  T* expected_ptr = expected_data.data();
  
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), [=](auto& x){
    T expected = T{0};
    success_ptr[0] = x.compare_exchange_strong(
        expected, T{12}, std::memory_order_relaxed, std::memory_order_relaxed);
    expected_ptr[0] = expected;

    expected = T{42};
    success_ptr[1] = x.compare_exchange_strong(
        expected, T{12}, std::memory_order_relaxed, std::memory_order_relaxed);
    expected_ptr[1] = expected;
  });

  BOOST_CHECK(data[0] == T{12});
  BOOST_CHECK(static_cast<bool>(success_ptr[0]) == false);
  BOOST_CHECK(static_cast<bool>(success_ptr[1]) == true);
  BOOST_CHECK(expected_ptr[0] == T{42});
  BOOST_CHECK(expected_ptr[1] == T{42});
}

namespace {

template<class T, class F>
void test_fetch_op(T expected_result, F f) {
  std::vector<std::atomic<T>> data(1);
  data[0] = T{42};
  std::vector<T> previous_value(1);
  T* previous_value_ptr = previous_value.data();
  
  std::for_each(std::execution::par_unseq, data.begin(), data.end(), [=](auto& x){
    *previous_value_ptr = f(x);
  });

  BOOST_CHECK(data[0] == expected_result);
  BOOST_CHECK(previous_value[0] == T{42});
}

}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_add, T, atomic_int_test_types::type) {
  test_fetch_op(T{44}, [](auto& x){ return x.fetch_add(T{2}, std::memory_order_relaxed);});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_sub, T, atomic_int_test_types::type) {
  test_fetch_op(T{40}, [](auto& x){ return x.fetch_sub(T{2}, std::memory_order_relaxed);});
}

/* C++ 26 
BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_max, T, atomic_test_types::type) {
  test_fetch_op(T{45}, [](auto& x){ return x.fetch_max(T{45}, std::memory_order_relaxed);});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_min, T, atomic_test_types::type) {
  test_fetch_op(T{41}, [](auto& x){ return x.fetch_min(T{41}, std::memory_order_relaxed);});
}
*/

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_or, T, atomic_int_test_types::type) {
  test_fetch_op(T{42}|T{15}, [](auto& x){ return x.fetch_or(T{15}, std::memory_order_relaxed);});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_and, T, atomic_int_test_types::type) {
  test_fetch_op(T{42}&T{15}, [](auto& x){ return x.fetch_and(T{15}, std::memory_order_relaxed);});
}

BOOST_AUTO_TEST_CASE_TEMPLATE(atomic_fetch_xor, T, atomic_int_test_types::type) {
  test_fetch_op(T{42}^T{30}, [](auto& x){ return x.fetch_xor(T{30}, std::memory_order_relaxed);});
}

BOOST_AUTO_TEST_SUITE_END()
