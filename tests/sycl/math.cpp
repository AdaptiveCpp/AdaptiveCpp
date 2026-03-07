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

#include <boost/type_traits/common_type.hpp>

#include <bitset>
#include <cmath>

BOOST_FIXTURE_TEST_SUITE(math_tests, reset_device_fixture)

// list of types classified as "genfloat" in the SYCL standard
using math_test_genfloats = boost::mp11::mp_list<
  float,
  // vec<T,1> is not genfloat according to SYCL 2020. It's unclear
  // if this is an oversight or intentional.
  // sycl::vec<float, 1>,
  sycl::vec<float, 2>,
  sycl::vec<float, 3>,
  sycl::vec<float, 4>,
  sycl::vec<float, 8>,
  sycl::vec<float, 16>,
  double,
  // sycl::vec<double, 1>,
  sycl::vec<double, 2>,
  sycl::vec<double, 3>,
  sycl::vec<double, 4>,
  sycl::vec<double, 8>,
  sycl::vec<double, 16>>;

namespace {

  template<typename DT, int D>
  using vec = sycl::vec<DT, D>;

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
  struct vector_dim {
    static constexpr int value = 0;
  };
  template<typename DT, int D>
  struct vector_dim<vec<DT, D>> {
    static constexpr int value = 1;
  };
  template<typename T>
  constexpr int vector_dim_v = vector_dim<T>::value;

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

  template<typename TARGET_DT, typename T>
  struct vector_coerce_elem {
    using type = TARGET_DT;
  };
  template<typename TARGET_DT, typename DT, int D>
  struct vector_coerce_elem<TARGET_DT, vec<DT, D>> {
    using type = vec<TARGET_DT, D>;
  };
  template<typename TARGET_DT, typename T>
  using vector_coerce_elem_t = typename vector_coerce_elem<TARGET_DT, T>::type;

  // utility functions for generic testing

  template <typename DT, int D>
  auto get_math_input(const vec<DT, 16> &v) {
    if constexpr(D==0) {
      return v.template swizzle<0>();
    } else if constexpr(D==1) {
      return vec<DT, 1>{v.template swizzle<0>()};
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

  // reference functions

  double ref_clamp(double a, double min, double max) { // in C++17 <algorithm>, remove on upgrade
    if(a < min) return min;
    if(a > max) return max;
    return a;
  }

  static constexpr double pi = 3.1415926535897932385;
  double ref_degrees(double v) {
    return 180.0/pi * v;
  }
  double ref_radians(double v) {
    return pi/180.0 * v;
  }

  double ref_mix(double x, double y, double a) {
    return x + (y - x) * a;
  }

  double ref_step(double edge, double x) {
    if(x < edge) return 0.0;
    return 1.0;
  }

  double ref_smoothstep(double edge0, double edge1, double x) {
    BOOST_REQUIRE(edge0 < edge1); // Standard: results are undefined if edge0 >= edge1
    if(x <= edge0) return 0.0;
    if(x >= edge1) return 1.0;
    double t = ref_clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
  }

  double ref_sign(double x) {
    if(x > 0.0) return 1.0;
    if(x < 0.0) return -1.0;
    if(std::isnan(x)) return 0.0;
    return x;
  }

  template<typename DT, int D>
  DT ref_dot(vec<DT, D> a, vec<DT, D> b) {
    DT ret = DT{0};
    for(int c = 0; c < D; ++c) {
      ret += comp(a, c) * comp(b, c);
    }
    return ret;
  }
  double ref_dot(double a, double b) {
    return a * b;
  }

  double ref_length(double v) {
    return std::abs(v);
  }
  template<typename DT>
  DT ref_length(vec<DT, 2> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y());
  }
  template<typename DT>
  DT ref_length(vec<DT, 3> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z());
  }
  template<typename DT>
  DT ref_length(vec<DT, 4> v) {
    return sqrt(v.x()*v.x() + v.y()*v.y() + v.z()*v.z() + v.w()*v.w());
  }

  template<typename T>
  auto ref_distance(T a, T b) {
    return ref_length(a - b);
  }

  template<typename T>
  auto ref_normalize(T v) {
    return v / ref_length(v);
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_ctz(T x) noexcept {
    if(x==0){return sizeof(T)*CHAR_BIT;}
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    int idx = 0;
    while(!bset[idx]){idx++;}
    return idx;
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_clz(T x) noexcept {
    if(x==0){return sizeof(T)*CHAR_BIT;}
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    int idx = 0;
    while(!bset[sizeof(T)*CHAR_BIT - idx -1]){idx++;}
    return idx;
  }

  template<class T, std::enable_if_t<std::is_integral_v<T>,int> = 0>
  inline T ref_popcount(T x) noexcept {
    std::bitset<sizeof(T)*CHAR_BIT> bset(x);
    return bset.count();
  }

  double ref_acosh(double x) {
    return std::acosh(x);
  }

  double ref_asinh(double x) {
    return std::asinh(x);
  }

  double ref_atanh(double x) {
    return std::atanh(x);
  }

  double ref_cbrt(double x) {
    return std::cbrt(x);
  }

  double ref_erf(double x) {
    return std::erf(x);
  }

  double ref_erfc(double x) {
    return std::erfc(x);
  }

  double ref_logb(double x) {
    return std::logb(x);
  }

  double ref_nextafter(double x, double y) {
    return std::nextafter(x, y);
  }

  double ref_remainder(double x, double y) {
    return std::remainder(x, y);
  }

  double ref_maxmag(double x, double y) {
    double ax = std::abs(x), ay = std::abs(y);
    if (ax > ay) {
      return x;
    }
    if (ay > ax) {
      return y;
    }
    return std::fmax(x, y);
  }

  double ref_minmag(double x, double y) {
    double ax = std::abs(x), ay = std::abs(y);
    if (ax < ay) {
      return x;
    }
    if (ay < ax) {
      return y;
    }
    return std::fmin(x, y);
  }

  double ref_acospi(double x) {
    return std::acos(x) / pi;
  }

  double ref_asinpi(double x) {
    return std::asin(x) / pi;
  }

  double ref_atanpi(double x) {
    return std::atan(x) / pi;
  }

  double ref_atan2pi(double x, double y) {
    return std::atan2(x, y) / pi;
  }

  double ref_cospi(double x) {
    return std::cos(x * pi);
  }

  double ref_sinpi(double x) {
    return std::sin(x * pi);
  }

  double ref_lgamma(double x) {
    return std::lgamma(x);
  }

  double ref_tgamma(double x) {
    return std::tgamma(x);
  }

  int ref_lgamma_r_sign(double x) {
    return std::tgamma(x) >= 0.0 ? 1 : -1;
  }

  double ref_fract(double x) {
    return std::min(x - std::floor(x), std::nextafter(1.0, 0.0));
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_binary, T,
                              math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 8;

  // build inputs

  s::queue queue;

  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class math_binary, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::atan2(acc[0], acc[1]);
      acc[i++] = s::copysign(acc[0], acc[1]);
      acc[i++] = s::fmin(acc[0], acc[1]);
      acc[i++] = s::fmax(acc[0], acc[1]);
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      // This triggers ICE in nvc++, no workaround yet.
      acc[i++] = s::fmod(acc[0], acc[1]);
#endif
      acc[i++] = s::fdim(acc[0], acc[1]);
      acc[i++] = s::hypot(acc[0], acc[1]);
      acc[i++] = s::pow(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::atan2(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
      BOOST_TEST(comp(acc[i++], c) == std::copysign(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
      BOOST_TEST(comp(acc[i++], c) == std::fmin(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
      BOOST_TEST(comp(acc[i++], c) == std::fmax(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
#ifndef ACPP_LIBKERNEL_CUDA_NVCXX
      BOOST_TEST(comp(acc[i++], c) == std::fmod(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
#endif
      BOOST_TEST(comp(acc[i++], c) == std::fdim(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
      BOOST_TEST(comp(acc[i++], c) == std::hypot(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
      BOOST_TEST(comp(acc[i++], c) == std::pow(static_cast<double>(comp(acc[0], c)), static_cast<double>(comp(acc[1], c))));
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(common_functions, T,
    math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 23;
  // build inputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> buf{{FUN_COUNT + 2}};
  DT input_scalar = 3.5f;
  DT mix_input_1 = 0.5f;
  DT mix_input_2 = 0.8f;
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions
  // some of these are tested multiple times to ensure that all overloads are covered
  // (e.g. combinations of vec and scalar input)

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class common_functions, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fabs(acc[0]);
      acc[i++] = s::clamp(acc[0], acc[1], acc[1] + static_cast<DT>(10));
      acc[i++] = s::clamp(acc[0], input_scalar, static_cast<DT>(input_scalar + 10));
      acc[i++] = s::degrees(acc[0]);
      acc[i++] = s::fma(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mad(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], input_scalar);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::min(acc[0], input_scalar);
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_1});
      acc[i++] = s::mix(acc[0], acc[1], T{mix_input_2});
      acc[i++] = s::mix(acc[0], acc[1], mix_input_1);
      acc[i++] = s::radians(acc[0]);
      acc[i++] = s::step(acc[0], acc[1]);
      acc[i++] = s::step(input_scalar, acc[0]);
      acc[i++] = s::smoothstep(acc[0], acc[0] + static_cast<DT>(10), acc[1]);
      acc[i++] = s::smoothstep(input_scalar, input_scalar + 1, acc[0]);
      acc[i++] = s::sign(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), comp(acc[1], c), comp(acc[1], c) + 10));
      BOOST_TEST(comp(acc[i++], c) == ref_clamp(comp(acc[0], c), input_scalar, input_scalar + 10));
      BOOST_TEST(comp(acc[i++], c) == ref_degrees(comp(acc[0], c)));
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1));
      BOOST_TEST(comp(acc[i++], c) == std::fma(comp(acc[0], c), comp(acc[1], c), mix_input_1)); // mad
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), input_scalar));
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), input_scalar));
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1));
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_2));
      BOOST_TEST(comp(acc[i++], c) == ref_mix(comp(acc[0], c), comp(acc[1], c), mix_input_1));
      BOOST_TEST(comp(acc[i++], c) == ref_radians(comp(acc[0], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_step(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_step(input_scalar, comp(acc[0], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(comp(acc[0], c), comp(acc[0], c) + 10, comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_smoothstep(input_scalar, input_scalar + 1, comp(acc[0], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_sign(comp(acc[0], c)));
    }
  }
}

// some subset of types classified as "geninteger" in SYCL
using math_test_genints = boost::mp11::mp_list<
  int,
  sycl::vec<int, 2>,
  sycl::vec<int, 3>,
  sycl::vec<int, 16>,
  short,
  sycl::vec<short, 4>,
  unsigned char,
  sycl::vec<unsigned char, 3>,
  unsigned long,
  sycl::vec<unsigned long, 8>>;

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(builtin_int_basic, T, math_test_genints) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 6;

  // build inputs

  s::queue queue;
  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class builtin_int_basic, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::abs(acc[0]);
      acc[i++] = s::min(acc[0], acc[1]);
      acc[i++] = s::max(acc[0], acc[1]);
      acc[i++] = s::ctz(acc[0]);
      acc[i++] = s::clz(acc[0]);
      acc[i++] = s::popcount(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      if constexpr(std::is_signed<DT>::value)
        BOOST_TEST(comp(acc[i++], c) == std::abs(comp(acc[0], c)));
      else
        BOOST_TEST(comp(acc[i++], c) == comp(acc[0], c));
      BOOST_TEST(comp(acc[i++], c) == std::min(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == std::max(comp(acc[0], c), comp(acc[1], c)));
      BOOST_TEST(comp(acc[i++], c) == ref_ctz(comp(acc[0], c)));
      // It seems that certain LLVM/ROCm versions in CI miscompile this test in SMCP
      // mode. Temporarily disable in HIP SMCP. We still test with SSCP on AMD,
      // and with SMCP on non-AMD devices, including explicit multipass builds
      // were both CUDA and HIP are targeted in a single build.
#ifdef __ACPP_ENABLE_HIP_TARGET__
      bool enable_clz = true;//queue.get_device().get_backend() != s::backend::hip;
#else
      bool enable_clz = true;
#endif
      if(enable_clz){
        BOOST_TEST(comp(acc[i++], c) == ref_clz(comp(acc[0], c)));
      } else {
        i++;
      }
      BOOST_TEST(comp(acc[i++], c) == ref_popcount(comp(acc[0], c)));
    }
  }
}


// types allowed for the "cross" function
using math_test_crossinputs = boost::mp11::mp_list<
  sycl::vec<float, 3>,
  sycl::vec<float, 4>,
  sycl::vec<double, 3>,
  sycl::vec<double, 4>>;

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(geometric_cross, T, math_test_crossinputs) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 1;

  // build inputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for (int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric_cross, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::cross(acc[0], acc[1]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    int i = 2;
    const auto& res = acc[i++], a = acc[0], b = acc[1];
    BOOST_TEST(res.x() == a.y()*b.z() - a.z()*b.y());
    BOOST_TEST(res.y() == a.z()*b.x() - a.x()*b.z());
    BOOST_TEST(res.z() == a.x()*b.y() - a.y()*b.x());
    if(D==4) BOOST_TEST(comp(res,3) == DT{0});
  }
}

// type classes as per SYCL standard

using math_test_gengeofloats = boost::mp11::mp_list<
  float,
  sycl::vec<float, 2>,
  sycl::vec<float, 3>,
  sycl::vec<float, 4>>;

using math_test_gengeodoubles = boost::mp11::mp_list<
  double,
  sycl::vec<double, 2>,
  sycl::vec<double, 3>,
  sycl::vec<double, 4>>;

using math_test_gengeo = boost::mp11::mp_append<math_test_gengeofloats, math_test_gengeodoubles>;

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(geometric, T, math_test_gengeo) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 4;

  // build inputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::dot(acc[0], acc[1]);
      acc[i++] = s::length(acc[0]);
      acc[i++] = s::distance(acc[0], acc[1]);
      acc[i++] = s::normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto dot_ref_result = ref_dot(acc[0], acc[1]);
    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(dot_ref_result));
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result));
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result));
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)));
    }
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(fast_geometric, T, math_test_gengeofloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 3;

  // build inputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> buf{{FUN_COUNT + 2}};
  {
    auto acc = buf.template get_access<s::access::mode::write>();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<DT, 16> v2{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    acc[0] = get_math_input<DT, D>(v1);
    acc[1] = get_math_input<DT, D>(v2);
    for(int i = 2; i < FUN_COUNT + 2; ++i) {
      acc[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto acc = buf.template get_access<s::access::mode::read_write>(cgh);
    cgh.single_task<kernel_name<class fast_geometric, D, DT>>([=]() {
      int i = 2;
      acc[i++] = s::fast_length(acc[0]);
      acc[i++] = s::fast_distance(acc[0], acc[1]);
      acc[i++] = s::fast_normalize(acc[0]);
    });
  });

  // check results

  {
    auto acc = buf.template get_access<s::access::mode::read>();

    auto length_ref_result = ref_length(acc[0]);
    auto distance_ref_result = ref_distance(acc[0], acc[1]);
    auto normalize_ref_result = ref_normalize(acc[0]);
    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 2;
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(length_ref_result));
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(distance_ref_result));
      BOOST_TEST(comp(acc[i++], c) == static_cast<double>(comp(normalize_ref_result, c)));
    }
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_int, T,
                              math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  constexpr int FUN_COUNT = 1;

  // build inputs and allocate outputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();
    s::vec<DT, 16> v{17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0};
    inputs[0] = get_math_input<DT, D>(v);
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto inputs  = in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_genfloat_int, D, DT>>([=]() {
      outputs[0] = s::ldexp(inputs[0], 7);
    });
  });

  // check results

  {
    auto inputs  = in.get_host_access();
    auto outputs = out.get_host_access();

    for(int c = 0; c < std::max(D,1); ++c) {
      BOOST_TEST(comp(outputs[0], c) == std::ldexp(comp(inputs[0], c), 7));
    }
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_genfloat_genint, T,
                              math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  using IntType = s::detail::builtin_input_intlike_t<T>;

  constexpr int FUN_COUNT = 3;

  // build inputs and allocate outputs

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  s::buffer<T> float_in{{1}};
  s::buffer<IntType> int_in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto float_inputs = float_in.get_host_access();
    auto int_inputs = int_in.get_host_access();
    auto outputs = out.get_host_access();
    s::vec<DT, 16> v1{7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0};
    s::vec<int, 16> v2{17, -4, -2, 3, 7, -8, 9, -1, 17, -4, -2, 3, 7, -8, 9, -1};
    float_inputs[0] = get_math_input<DT, D>(v1);
    int_inputs[0] = get_math_input<int, D>(v2);
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler &cgh) {
    auto float_inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto int_inputs = int_in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_genfloat_genint, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::ldexp(float_inputs[0], int_inputs[0]);
      outputs[i++] = s::pown(float_inputs[0], int_inputs[0]);
      outputs[i++] = s::rootn(s::fabs(float_inputs[0]), int_inputs[0]);
    });
  });

  // check results

  {
    auto float_inputs = float_in.get_host_access();
    auto int_inputs = int_in.get_host_access();
    auto outputs = out.get_host_access();

    for(int c = 0; c < std::max(D,1); ++c) {
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == std::ldexp(comp(float_inputs[0], c), comp(int_inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::pow(comp(float_inputs[0], c), comp(int_inputs[0], c)));
      BOOST_TEST(comp(outputs[i++], c) == std::pow(std::fabs(comp(float_inputs[0], c)), 1./comp(int_inputs[0], c)));
    }
  }
}

BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_unary_extra, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;
  using IntType = vector_coerce_elem_t<int, T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  constexpr int FUN_COUNT = 5; // acosh, asinh, atanh, cbrt, logb

  s::buffer<T> float_in{{2}};
  s::buffer<T> out{{FUN_COUNT}};
  s::buffer<IntType> ilogb_out{{1}};
  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      1.5, 2.0, 1.25, 3.0, 1.1, 1.7, 2.5, 1.3, 1.5, 2.0, 1.25, 3.0, 1.1, 1.7, 2.5, 1.3});
    inputs[1] = get_math_input<DT, D>(s::vec<DT, 16>{
      0.5, -0.3,  0.75, -0.1, 0.6, -0.5, 0.2, -0.8, 0.5, -0.3, 0.75, -0.1, 0.6, -0.5, 0.2, -0.8});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template  get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    auto ilogb = ilogb_out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_unary_extra, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::acosh(inputs[0]);
      outputs[i++] = s::asinh(inputs[1]);
      outputs[i++] = s::atanh(inputs[1]);
      outputs[i++] = s::cbrt(inputs[1]);
      outputs[i++] = s::logb(inputs[1]);
      ilogb[0] = s::ilogb(inputs[1]);
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    auto ilogb = ilogb_out.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x0 = static_cast<double>(comp(inputs[0], c));
      double x1 = static_cast<double>(comp(inputs[1], c));
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == ref_acosh(x0));
      BOOST_TEST(comp(outputs[i++], c) == ref_asinh(x1));
      BOOST_TEST(comp(outputs[i++], c) == ref_atanh(x1));
      BOOST_TEST(comp(outputs[i++], c) == ref_cbrt(x1));
      BOOST_TEST(comp(outputs[i++], c) == ref_logb(x1));
      BOOST_TEST(comp(ilogb[0], c) == std::ilogb(comp(inputs[1], c)));
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_binary_extra, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  constexpr int FUN_COUNT = 4; // nextafter, remainder, maxmag, minmag

  s::buffer<T> float_in{{2}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0});
    inputs[1] = get_math_input<DT, D>(s::vec<DT, 16>{
      17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0, 17.0, -4.0, -2.0, 3.0, 7.0, -8.0, 9.0, -1.0});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_binary_extra, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::nextafter(inputs[0], inputs[1]);
      outputs[i++] = s::remainder(inputs[0], inputs[1]);
      outputs[i++] = s::maxmag(inputs[0], inputs[1]);
      outputs[i++] = s::minmag(inputs[0], inputs[1]);
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x = static_cast<double>(comp(inputs[0], c));
      double y = static_cast<double>(comp(inputs[1], c));
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == ref_nextafter(x, y));
      BOOST_TEST(comp(outputs[i++], c) == ref_remainder(x, y));
      BOOST_TEST(comp(outputs[i++], c) == ref_maxmag(x, y));
      BOOST_TEST(comp(outputs[i++], c) == ref_minmag(x, y));
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_pi_functions, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  constexpr int FUN_COUNT = 6; // acospi, asinpi, atanpi, atan2pi, cospi, sinpi

  s::buffer<T> float_in{{2}};
  s::buffer<T> out{{FUN_COUNT}};
  {
    auto inputs  = float_in.get_host_access();
    auto outputs = out.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      0.4, -0.3, 0.75, -0.1, 0.6, -0.4, 0.2, -0.8, 0.4, -0.3, 0.75, -0.1, 0.6, -0.4, 0.2, -0.8});
    inputs[1] = get_math_input<DT, D>(s::vec<DT, 16>{
      7.0, -4.0, 2.0, 3.0, -5.0, 1.0, -3.0, 0.5, 7.0, -4.0, 2.0, 3.0, -5.0, 1.0, -3.0, 0.5});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_pi_functions, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::acospi(inputs[0]);
      outputs[i++] = s::asinpi(inputs[0]);
      outputs[i++] = s::atanpi(inputs[0]);
      outputs[i++] = s::atan2pi(inputs[0], inputs[1]);
      outputs[i++] = s::cospi(inputs[0]);
      outputs[i++] = s::sinpi(inputs[0]);
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x = static_cast<double>(comp(inputs[0], c));
      double y = static_cast<double>(comp(inputs[1], c));
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == ref_acospi(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_asinpi(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_atanpi(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_atan2pi(x, y));
      BOOST_TEST(comp(outputs[i++], c) == ref_cospi(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_sinpi(x));
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0007))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_erf_erfc, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  s::buffer<T> float_in{{1}};
  s::buffer<T> erf_out {{1}};
  s::buffer<T> erfc_out{{1}};
  {
    auto inputs = float_in.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 2.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 2.0});
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto eout = erf_out.template get_access<s::access::mode::write>(cgh);
    auto ecout = erfc_out.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_erf_erfc_detailed, D, DT>>([=]() {
      eout[0] = s::erf(inputs[0]);
      ecout[0] = s::erfc(inputs[0]);
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto eout = erf_out.get_host_access();
    auto ecout = erfc_out.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x = static_cast<double>(comp(inputs[0], c));
      BOOST_TEST(comp(eout[0], c) == ref_erf(x));
      BOOST_TEST(comp(ecout[0], c) == ref_erfc(x));
      // erf(x) + erfc(x) == 1
      BOOST_TEST(comp(eout[0], c) + comp(ecout[0], c) == 1.0);
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_gamma, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;
  using IntType = vector_coerce_elem_t<int, T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  constexpr int FUN_COUNT = 3; // lgamma, tgamma, lgamma_r

  s::buffer<T> float_in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  s::buffer<IntType> lgr_sgn {{1}};
  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      0.5, 1.5, 2.5, -0.5, 3.0, -1.5, 4.0, -2.1, 0.5, 1.5, 2.5, -0.5, 3.0, -1.5, 4.0, -2.1});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    auto sgn = lgr_sgn.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_gamma, D, DT>>([=]() {
      int i = 0;
      outputs[i++] = s::lgamma(inputs[0]);
      outputs[i++] = s::tgamma(inputs[0]);
      IntType s;
      outputs[i++] = s::lgamma_r(inputs[0], &s);
      sgn[0] = s;
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    auto sgn = lgr_sgn.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x = static_cast<double>(comp(inputs[0], c));
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == ref_lgamma(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_tgamma(x));
      BOOST_TEST(comp(outputs[i++], c) == ref_lgamma(x));
      BOOST_TEST(comp(sgn[0], c) == ref_lgamma_r_sign(x));
    }
  }
}


BOOST_TEST_DECORATOR(*boost::unit_test::tolerance(0.0001))
BOOST_AUTO_TEST_CASE_TEMPLATE(math_out_params, T, math_test_genfloats) {

  constexpr int D = vector_length_v<T>;
  using DT = vector_elem_t<T>;
  using IntType = vector_coerce_elem_t<int, T>;

  namespace s = sycl;

  s::queue queue;
  if constexpr(std::is_same_v<DT, double>) {
    if (!queue.get_device().has(sycl::aspect::fp64)) {
      BOOST_TEST_MESSAGE("Skipping test for double since device has no fp64 support");
      return;
    }
  }

  // build inputs and allocate outputs

  // out[0]=frexp mantissa, out[1]=modf frac, out[2]=modf ipart, out[3]=fract frac
  constexpr int FUN_COUNT = 4;

  s::buffer<T> float_in{{1}};
  s::buffer<T> out{{FUN_COUNT}};
  s::buffer<IntType> frexp_exp{{1}};
  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    inputs[0] = get_math_input<DT, D>(s::vec<DT, 16>{
      0.75, 1.5, -2.25, 0.5, 3.125, -0.375, 1.0, 0.25,
      0.75, 1.5, -2.25, 0.5, 3.125, -0.375, 1.0, 0.25});
    for(int i = 0; i < FUN_COUNT; ++i) {
      outputs[i] = T{DT{0}};
    }
  }

  // run functions

  queue.submit([&](s::handler& cgh) {
    auto inputs = float_in.template  get_access<s::access::mode::read>(cgh);
    auto outputs = out.template get_access<s::access::mode::write>(cgh);
    auto exp = frexp_exp.template get_access<s::access::mode::write>(cgh);
    cgh.single_task<kernel_name<class math_out_params, D, DT>>([=]() {
      int i = 0;
      IntType frexp_exp_val;
      outputs[i++] = s::frexp(inputs[0], &frexp_exp_val);
      exp[0] = frexp_exp_val;

      T modf_ipart;
      outputs[i++] = s::modf(inputs[0], &modf_ipart);
      outputs[i++] = modf_ipart;

      T fract_ipart;
      outputs[i++] = s::fract(inputs[0], &fract_ipart);
    });
  });

  // check results

  {
    auto inputs = float_in.get_host_access();
    auto outputs = out.get_host_access();
    auto exp = frexp_exp.get_host_access();
    for(int c = 0; c < std::max(D, 1); ++c) {
      double x = static_cast<double>(comp(inputs[0], c));
      int ref_frexp_exp; double ref_frexp_m = std::frexp(x, &ref_frexp_exp);
      double ref_modf_ipart; double ref_modf_frac = std::modf(x, &ref_modf_ipart);
      int i = 0;
      BOOST_TEST(comp(outputs[i++], c) == ref_frexp_m);
      BOOST_TEST(comp(exp[0], c) == ref_frexp_exp);
      BOOST_TEST(comp(outputs[i++], c) == ref_modf_frac);
      BOOST_TEST(comp(outputs[i++], c) == ref_modf_ipart);
      BOOST_TEST(comp(outputs[i++], c) == ref_fract(x));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END() // NOTE: Make sure not to add anything below this line
