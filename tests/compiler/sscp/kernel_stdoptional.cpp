// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s
// UNSUPPORTED: cuda || hip || ocl || ze

#include <iostream>
#include <optional>
#include <string>
#include <sycl/sycl.hpp>
int main() {
  sycl::queue q;
  int* data = sycl::malloc_shared<int>(1, q);
  q.single_task([=](){
    try{
      std::optional<int> opt;
      *data = opt.value();
    }
    catch(std::bad_optional_access& e){
      *data = 42;
    }
  });
  q.wait();
  //CHECK: ExceptionToAssertionPass: Exception in Device Code
  return 0;
}

