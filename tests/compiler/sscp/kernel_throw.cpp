// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic
// RUN: %hcfi %s.hcf | FileCheck %s 
// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic -O3
// RUN: %hcfi %s.hcf | FileCheck %s 
// RUN: %acpp -mllvm -acpp-sscp-emit-hcf %s -o %t --acpp-targets=generic -g
// RUN: %hcfi %s.hcf | FileCheck %s
#include <sycl/sycl.hpp>
#include <iostream>
int main() {
  sycl::queue q;
  int* data = sycl::malloc_shared<int>(1, q);
  q.single_task([=]{
      try{throw data[0];}
      catch(int){*data=42;}}).wait();
 // CHECK-NOT: CXATHROWHIT
  return 0;
}

