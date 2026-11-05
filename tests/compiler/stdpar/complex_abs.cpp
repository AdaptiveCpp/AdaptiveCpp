// RUN: %acpp %s -o %t --acpp-targets=generic -O0 \
// RUN:   --acpp-stdpar --acpp-stdpar-unconditional-offload -D_GLIBCXX_USE_TBB_PAR_BACKEND=0
// RUN: rm -rf %t.appdb %t.ll
// RUN: env ACPP_APPDB_DIR=%t.appdb ACPP_S2_DUMP_IR_INPUT=%t.ll %t
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll
// RUN: %acpp %s -o %t --acpp-targets=generic -O3 \
// RUN:   --acpp-stdpar --acpp-stdpar-unconditional-offload -D_GLIBCXX_USE_TBB_PAR_BACKEND=0
// RUN: rm -rf %t.appdb %t.ll
// RUN: env ACPP_APPDB_DIR=%t.appdb ACPP_S2_DUMP_IR_INPUT=%t.ll %t
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.ll

// Unconditional offload keeps this small workload out of the host fallback.
// A fresh private cache forces JIT; inspect IR before host libm can resolve cabs.
// IR-NOT: declare {{.*}}@cabs{{f?}}(
// IR: stage: input
// IR-NOT: declare {{.*}}@cabs{{f?}}(
// IR: define {{.*}}__acpp_sscp_kernel
// IR-NOT: declare {{.*}}@cabs{{f?}}(

#include <algorithm>
#include <cmath>
#include <complex>
#include <execution>
#include <iostream>
#include <limits>
#include <vector>
#include <hipSYCL/std/stdpar/detail/sycl_glue.hpp>

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
bool test() {
  const T inf = std::numeric_limits<T>::infinity();
  const T nan = std::numeric_limits<T>::quiet_NaN();
  // Avoid dependence on denormal support, while still detecting naive squaring.
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
  std::vector<T> output(input.size(), T(-1));

  // Heap-backed input reaches the offloaded kernel as runtime loads.
  std::transform(std::execution::par_unseq, input.begin(), input.end(),
                 output.begin(), [](const std::complex<T> &z) {
                   return std::abs(z);
                 });

  bool ok = true;
  for(std::size_t i = 0; i < input.size(); ++i) {
    const T expected = std::hypot(input[i].real(), input[i].imag());
    if(!matches(output[i], expected)) {
      std::cerr << "complex abs (" << sizeof(T) * 8 << "-bit), case " << i
                << ": got " << output[i] << ", expected " << expected << '\n';
      ok = false;
    }
  }
  return ok;
}

int main() {
  bool ok = test<float>();
  // Query the actual stdpar queue, not a separately selected SYCL device.
  auto &q = hipsycl::stdpar::detail::single_device_dispatch::get_queue();
  if(q.get_device().has(hipsycl::sycl::aspect::fp64))
    ok = test<double>() && ok;
  return ok ? 0 : 1;
}
