// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s
#include <sycl/sycl.hpp>
#include <iostream>
int main() {
  sycl::queue q;
  int* data = sycl::malloc_shared<int>(1, q);
  q.single_task([=]{
      try{throw data[0];}
      catch(int){*data=42;}}).wait();
  //CHECK: ExceptionToAssertionPass: Exception in Device Code
  return 0;
}

