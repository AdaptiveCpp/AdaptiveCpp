// RUN: %acpp %s -o %t --acpp-targets=generic -O0
// RUN: rm -rf %t.appdb %t.ll
// RUN: env ACPP_APPDB_DIR=%t.appdb ACPP_S2_DUMP_IR_INPUT=%t.ll %t
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: rm -rf %t.appdb %t.ll
// RUN: env ACPP_APPDB_DIR=%t.appdb ACPP_S2_DUMP_IR_INPUT=%t.ll %t
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll

// A fresh private cache forces JIT and prevents stale IR from passing the test.
// Check before backend linking: host libm could otherwise hide unresolved cabs.
// IR-NOT: declare {{.*}}@cabs{{f?}}(
// IR: stage: input
// IR-NOT: declare {{.*}}@cabs{{f?}}(
// IR: define {{.*}}__acpp_sscp_kernel
// IR-NOT: declare {{.*}}@cabs{{f?}}(

#include "common.hpp"
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <vector>

// Exercise the C builtins even when the standard library implements abs via hypot.
float builtin_abs(float real, float imag) {
  return __builtin_cabsf(__builtin_complex(real, imag));
}

double builtin_abs(double real, double imag) {
  return __builtin_cabs(__builtin_complex(real, imag));
}

template<class T>
bool matches(T actual, T expected) {
  if(std::isnan(expected))
    return std::isnan(actual);
  if(std::signbit(actual))
    return false;
  if(std::isinf(expected) || expected == T(0))
    return actual == expected;
  return std::isfinite(actual) &&
         std::abs((actual - expected) / expected) <=
             T(16) * std::numeric_limits<T>::epsilon();
}

template<class T>
bool test(sycl::queue &q) {
  const T inf = std::numeric_limits<T>::infinity();
  const T nan = std::numeric_limits<T>::quiet_NaN();
  // Both inputs and results are normal and finite, but squaring over/underflows.
  const T large = std::numeric_limits<T>::max() / T(4);
  const T small = std::numeric_limits<T>::min() * T(4);
  const std::vector<std::complex<T>> input{
      {3, 4}, {-3, 4}, {3, -4}, {-3, -4}, {0, 5}, {-7, 0},
      {T(0), T(0)}, {-T(0), T(0)}, {T(0), -T(0)}, {-T(0), -T(0)},
      {large, large}, {large, -large / T(2)}, {large, small}, {small, large},
      {small, small}, {small, -small / T(2)},
      {inf, 1}, {1, -inf}, {inf, -inf},
      {nan, 1}, {1, nan}, {nan, nan},
      {inf, nan}, {nan, inf}, {-inf, nan}, {nan, -inf}};
  const auto n = input.size();
  std::vector<T> output(2 * n, T(-1));
  auto *device_input = sycl::malloc_device<std::complex<T>>(n, q);
  auto *device_output = sycl::malloc_device<T>(output.size(), q);
  q.memcpy(device_input, input.data(), n * sizeof(std::complex<T>)).wait_and_throw();
  q.memcpy(device_output, output.data(), output.size() * sizeof(T)).wait_and_throw();

  // Load from device memory rather than folding a complex literal in the kernel.
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
    auto z = device_input[i[0]];
    device_output[i[0]] = std::abs(z);
    device_output[n + i[0]] = builtin_abs(z.real(), z.imag());
  }).wait_and_throw();
  q.memcpy(output.data(), device_output, output.size() * sizeof(T)).wait_and_throw();
  sycl::free(device_input, q);
  sycl::free(device_output, q);

  bool ok = true;
  for(std::size_t i = 0; i < output.size(); ++i) {
    const auto z = input[i % n];
    const T expected = std::hypot(z.real(), z.imag());
    if(!matches(output[i], expected)) {
      std::cerr << (i < n ? "std::abs" : "builtin cabs") << " ("
                << sizeof(T) * 8 << "-bit), case " << i % n
                << ": got " << output[i] << ", expected " << expected << '\n';
      ok = false;
    }
  }
  return ok;
}

int main() {
  auto q = get_queue();
  bool ok = test<float>(q);
  if(q.get_device().has(sycl::aspect::fp64))
    ok = test<double>(q) && ok;
  return ok ? 0 : 1;
}
