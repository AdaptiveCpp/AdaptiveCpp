// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>

#include <sycl/sycl.hpp>
#include "common.hpp"

unsigned run_cpuid() {
#ifdef __x86_64__
  unsigned a = 0x1, b, c, d;
  asm volatile("cpuid"
               : "=a"(a), "=b"(b), "=c"(c), "=d"(d)
               : "a"(a), "b"(b), "c"(c), "d"(d));
  return a;
#else
  return 0;
#endif
}

int main()
{
  sycl::queue q = get_queue();

  unsigned* data = sycl::malloc_device<unsigned>(4, q);
  for(int i = 0; i < 4; ++i)
    q.memset(data + i, 0, sizeof(unsigned));
  q.wait();

  q.parallel_for(sycl::range{1024}, [=](auto idx) {
     __acpp_if_target_sscp(
        sycl::AdaptiveCpp_jit::compile_if(
            __acpp_sscp_jit_reflect_compiler_backend() ==
                sycl::AdaptiveCpp_jit::compiler_backend::host,
            [&]() { *data = run_cpuid(); });

     );
   }).wait();

  int result;
  q.memcpy(&result, data, sizeof(int)).wait();

  // CHECK: 1
#ifdef __x86_64__
  if(q.get_device().get_backend() != sycl::backend::omp) {
    std::cout << 1 << std::endl;  
  } else {
    std::cout << (result > 1) << std::endl;
  }
#else
  std::cout << 1 << std::endl;
#endif

   sycl::free(data,q);
}
