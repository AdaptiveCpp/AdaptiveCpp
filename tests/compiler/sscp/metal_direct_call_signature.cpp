// REQUIRES: metal
// RUN: %acpp %s -x c++ %S/metal_direct_call_signature_impl.cpp.in -x none -o %t.o0 --acpp-targets=generic -O0
// RUN: %t.o0 | FileCheck %s
// RUN: %acpp %s -x c++ %S/metal_direct_call_signature_impl.cpp.in -x none -o %t.o3 --acpp-targets=generic -O3
// RUN: %t.o3 | FileCheck %s

#include "metal_direct_call_signature.hpp"

#include <iostream>

struct CallerTwoFloat {
  float First;
  float Second;
};

int main() {
  sycl::queue Queue;
  float *Output = sycl::malloc_shared<float>(2, Queue);

  Queue.single_task([=]() {
    CallerTwoFloat KeepTypeAlive{3.0f, 4.0f};
    const auto [First, Second] = makeAggregatePair(10.0f);
    Output[0] = First + Second + 0.0f * KeepTypeAlive.First;
    Output[1] = consumeAggregatePair(40.0f);
  }).wait();

  // CHECK: 23 83
  std::cout << Output[0] << ' ' << Output[1] << '\n';
  sycl::free(Output, Queue);
}
