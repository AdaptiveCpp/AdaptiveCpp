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

#include <boost/mpl/joint_view.hpp>

#include <cmath>

BOOST_FIXTURE_TEST_SUITE(rel_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using rel_test_genfloats = boost::mpl::list<
  float,
  // vec<T,1> is not genfloat according to SYCL 2020. It's unclear
  // if this is an oversight or intentional.
  //cl::sycl::vec<float, 1>
  cl::sycl::vec<float, 2>,
  cl::sycl::vec<float, 3>,
  cl::sycl::vec<float, 4>,
  cl::sycl::vec<float, 8>,
  cl::sycl::vec<float, 16>,
  double,
  //cl::sycl::vec<double, 1>,
  cl::sycl::vec<double, 2>,
  cl::sycl::vec<double, 3>,
  cl::sycl::vec<double, 4>,
  cl::sycl::vec<double, 8>,
  cl::sycl::vec<double, 16>>;



namespace {

  template<typename DT, int D>
  using vec = cl::sycl::vec<DT, D>;

  // utility type traits for generic testing

  template<typename T>
  struct vector_length {
    static constexpr int value = 0;
  };
  template<typename DT, int D>
  struct vector_length<vec<DT, D>> {
    static constexpr int value = D;
  };
  template<typename T>
  constexpr int vector_length_v = vector_length<T>::value;

  template<typename T>
  struct vector_elem {
    using type = T;
  };
  template<typename DT, int D>
  struct vector_elem<vec<DT, D>> {
    using type = DT;
  };
  template<typename T>
  using vector_elem_t = typename vector_elem<T>::type;

  // utility functions for generic testing

  template<typename DT, int D>
  auto get_subvector(const vec<DT, 16> &v) {
    if constexpr(D==0) {
      return v.template swizzle<0>();
    } else if constexpr(D==2) {
      return vec<DT, 2>{v.template swizzle<0,1>()};
    } else if constexpr(D==3) {
      return vec<DT, 3>{v.template swizzle<0,1,2>()};
    } else if constexpr(D==4) {
      return vec<DT, 4>{v.template swizzle<0,1,2,3>()};
    } else if constexpr(D==8) {
      return vec<DT, 8>{v.template swizzle<0,1,2,3,4,5,6,7>()};
    } else if constexpr(D==16) {
      return v;
    }
  }

  template<typename T>
  auto comp(T v, size_t idx) {
    assert(idx == 0);
    return v;
  }
  template<typename DT, int D>
  auto comp(vec<DT, D> v, size_t idx) {
    assert(idx < D);
    return v[idx];
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(rel_genfloat_unary, T,
                              rel_test_genfloats::type) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = cl::sycl;

  using IntType = typename std::conditional_t<std::is_same_v<DT, float>, int32_t, int64_t>;
  using OutType = typename s::detail::builtin_type_traits<T>::template alternative_data_type<IntType>;

  constexpr int FUN_COUNT = 5;

  // build inputs and allocate outputs

  s::queue queue;
  s::buffer<T> in{{1}};
  s::buffer<OutType> out{{FUN_COUNT}};
  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();
    s::vec<DT, 16> v{NAN, INFINITY, INFINITY - INFINITY,
		     0.0, 0.0/0.0, 1.0/0.0, sqrt(-1),
		     std::numeric_limits<float>::min(),
		     std::numeric_limits<float>::denorm_min(),
		     std::numeric_limits<double>::min(),
		     std::numeric_limits<double>::denorm_min(),
		     -1.0, 17.0, -4.0, -2.0, 3.0};
    inputs[0] = get_subvector<DT, D>(v);
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = OutType{IntType{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto inputs  = in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class rel_unary, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::isfinite(inputs[0]);
      outputs[i++] = s::isinf(inputs[0]);
      outputs[i++] = s::isnan(inputs[0]);
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      outputs[i++] = s::isnormal(inputs[0]);
#endif
      outputs[i++] = s::signbit(inputs[0]);
    });
  });

  // check results

  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == std::isfinite(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::isinf(comp(inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::isnan(comp(inputs[0], c)));
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      BOOST_TEST(comp(outputs[i++], c) == std::isnormal(comp(inputs[0], c)));
#endif
      BOOST_TEST(comp(outputs[i++], c) == std::signbit(comp(inputs[0], c)));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
