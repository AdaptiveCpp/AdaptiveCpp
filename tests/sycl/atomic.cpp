/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay
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


#include "sycl_test_suite.hpp"
#include <cstddef>
#include <type_traits>

using namespace cl;

BOOST_FIXTURE_TEST_SUITE(atomic_tests, reset_device_fixture)

template<class T, class IntT>
T int_to_t(IntT i) {
  T result{};

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<T>(i);
  } else {
    result = static_cast<T>(i);
  }

  return result;
}

template<class T>
unsigned long long t_to_int(T x) {
  unsigned long long result = 0ull;

  if constexpr(std::is_pointer_v<T>) {
    result = reinterpret_cast<unsigned long long>(x);
  } else {
    result = static_cast<unsigned long long>(x);
  }
  
  return result;
}

using exchange_test_types =
    boost::mpl::list<int, unsigned int, long, unsigned long, long long,
                     unsigned long long, float, double, int *>;
// mainly compile test
BOOST_AUTO_TEST_CASE_TEMPLATE(load_store_exchange, Type,
                              exchange_test_types::type) {
  sycl::queue q;

  Type initial = int_to_t<Type>(0);
  sycl::buffer<Type> b{&initial, sycl::range{1}};

  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.single_task([=]() {
      sycl::atomic_ref<Type, sycl::memory_order::relaxed,
                       sycl::memory_scope::device>
          r{acc[0]};
      unsigned long long x = t_to_int(r.load());

      // There seems to be a bug in clang related to incorrect
      // code generation at least on CUDA for floating point
      // stores.
      if constexpr(!std::is_floating_point_v<Type>)
        r.store(int_to_t<Type>(x+2));
      
      r.exchange(int_to_t<Type>(x+1));
    });
  });
  sycl::host_accessor hacc{b};

  BOOST_CHECK(t_to_int(hacc[0]) == 1);
}


template <class T, class AtomicTester,
          class Verifier>
void atomic_device_reduction_test(AtomicTester t, Verifier v) {
  
  sycl::queue q;

  const std::size_t size = 2048;
  sycl::buffer<T> b{sycl::range{size}};
  {
    sycl::host_accessor hacc{b};
    for(std::size_t i = 0; i < size; ++i) {
      hacc[i] = int_to_t<T>(i);
    }
  }
  q.submit([&](sycl::handler& cgh){
    sycl::accessor acc{b, cgh, sycl::read_write};
    cgh.parallel_for(sycl::range{size}, [=](sycl::id<1> idx){
      if(idx.get(0) != 0) {
        sycl::atomic_ref<T, sycl::memory_order::relaxed,
                        sycl::memory_scope::device> r{acc[0]};
        if constexpr(std::is_pointer_v<T>) {
          t(r, reinterpret_cast<std::ptrdiff_t>(acc[idx]));
        } else {
          t(r, acc[idx]);
        }
      }
    });
  });
  {
    sycl::host_accessor hacc{b};
    T expected = int_to_t<T>(0);
    for(std::size_t i = 1; i < size; ++i) {
      if constexpr(std::is_pointer_v<T>) {
        v(expected, static_cast<std::ptrdiff_t>(i));
      } else {
        v(expected, int_to_t<T>(i));
      }
    }
    BOOST_CHECK(expected == hacc[0]);
  }
}

BOOST_AUTO_TEST_CASE(fetch_op) {

  auto fetch_add = [](auto& atomic, auto x) {
    return atomic.fetch_add(x);
  };

  auto fetch_sub = [](auto& atomic, auto x) {
    return atomic.fetch_sub(x);
  };

  auto fetch_and = [](auto& atomic, auto x) {
    return atomic.fetch_and(x);
  };

  auto fetch_or = [](auto& atomic, auto x) {
    return atomic.fetch_or(x);
  };

  auto fetch_xor = [](auto& atomic, auto x) {
    return atomic.fetch_xor(x);
  };

  auto fetch_min = [](auto& atomic, auto x) {
    return atomic.fetch_min(x);
  };

  auto fetch_max = [](auto& atomic, auto x) {
    return atomic.fetch_max(x);
  };


  auto fetch_add_verifier = [](auto& v, auto x) {
    v += x;
  };

  auto fetch_sub_verifier = [](auto& v, auto x) {
    v -= x;
  };

  auto fetch_and_verifier = [](auto& v, auto x) {
    v &= x;
  };

  auto fetch_or_verifier = [](auto& v, auto x) {
    v |= x;
  };

  auto fetch_xor_verifier = [](auto& v, auto x) {
    v ^= x;
  };

  auto fetch_min_verifier = [](auto& v, auto x) {
    v = std::min(v,x);
  };

  auto fetch_max_verifier = [](auto& v, auto x) {
    v = std::max(v,x);
  };

#define HIPSYCL_ATOMIC_REF_INTEGER_TEST(Tester, Verifier)                      \
  atomic_device_reduction_test<int>(Tester, Verifier);                         \
  atomic_device_reduction_test<unsigned int>(Tester, Verifier);                \
  atomic_device_reduction_test<long>(Tester, Verifier);                        \
  atomic_device_reduction_test<unsigned long>(Tester, Verifier);               \
  atomic_device_reduction_test<long long>(Tester, Verifier);                   \
  atomic_device_reduction_test<unsigned long long>(Tester, Verifier);

#define HIPSYCL_ATOMIC_REF_FP_TEST(Tester, Verifier)                           \
  atomic_device_reduction_test<float>(Tester, Verifier);                       \
  atomic_device_reduction_test<double>(Tester, Verifier);

#define HIPSYCL_ATOMIC_REF_PTR_TEST(Tester, Verifier)                          \
  atomic_device_reduction_test<int *>(Tester, Verifier);

#ifndef HIPSYCL_LIBKERNEL_CUDA_NVCXX

  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_or, fetch_or_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_xor, fetch_xor_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_and, fetch_and_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_INTEGER_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_sub, fetch_sub_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_min, fetch_min_verifier);
  HIPSYCL_ATOMIC_REF_FP_TEST(fetch_max, fetch_max_verifier);

  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_add, fetch_add_verifier);
  HIPSYCL_ATOMIC_REF_PTR_TEST(fetch_sub, fetch_sub_verifier);

#endif
}

BOOST_AUTO_TEST_SUITE_END()
