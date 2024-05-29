// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -fsanitize=undefined
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -fsanitize=address
// RUN: %t | FileCheck %s

#include <iostream>
#include "hipSYCL/glue/reflection.hpp"

void myfunction1() {}
void myfunction2() {}

int main() {
  hipsycl::glue::reflection::enable_function_symbol_reflection(&myfunction1);

  const char* name1 = hipsycl::glue::reflection::resolve_function_name(&myfunction1);
  const char* name2 = hipsycl::glue::reflection::resolve_function_name(&myfunction2);

  // CHECK: _Z11myfunction1v
  std::cout << (name1 ? name1 : "nullptr") << std::endl;
  // CHECK: nullptr
  std::cout << (name2 ? name2 : "nullptr") << std::endl;
}
