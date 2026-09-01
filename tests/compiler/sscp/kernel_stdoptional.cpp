// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic
// RUN: %hcfi %s.hcf | FileCheck %s 
// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic -O3
// RUN: %hcfi %s.hcf | FileCheck %s 
// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic -g
// RUN: %hcfi %s.hcf | FileCheck %s

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
      *data = 424242;
    }
  });
  q.wait();
  std::cout << *data << std::endl;
  //CHECK-NOT: CXATHROWHIT
  return 0;
}

