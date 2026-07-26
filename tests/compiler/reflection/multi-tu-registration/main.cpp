// RUN: %acpp %s %S/other.cpp -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s %S/other.cpp -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s %S/other.cpp -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include "hipSYCL/glue/reflection.hpp"

bool other_tu_resolves();

int main() {
  // CHECK: 1
  std::cout << other_tu_resolves() << std::endl;
}
