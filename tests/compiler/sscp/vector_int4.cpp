// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <sycl/sycl.hpp>
#include "common.hpp"

static inline sycl::int4 loadInt4(sycl::global_ptr<const int> input, int index) {
  sycl::int4 value;
  value.load(index, input);
  return value;
}

static inline int process_interaction(sycl::global_ptr<const int> iatoms,
                                      sycl::global_ptr<const float> params,
                                      int tid) {
  sycl::int4 data = loadInt4(iatoms, tid);
  int type = data[0];
  int ai   = data[1];
  int aj   = data[2];
  int ak   = data[3];
  float p = params[type];
  return static_cast<int>(p) + ai + aj + ak;
}

int main() {
  sycl::queue q = get_queue();
  constexpr int N = 64;

  int* iatoms = sycl::malloc_device<int>(N * 4, q);
  float* params = sycl::malloc_device<float>(4, q);
  int* result = sycl::malloc_device<int>(N, q);

  int init[4] = {
                  0, // type
                  1, // ai
                  2, // aj
                  3  // ak
                };
  for (int i = 0; i < N; ++i) {
    q.memcpy(iatoms + i * 4, init, sizeof(init));
  }

  float p_init = 4.0f;
  q.memcpy(params, &p_init, sizeof(float));
  q.wait();

  q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> id) {
    int tid = id[0];
    sycl::global_ptr<const int> iatoms_ptr(iatoms);
    sycl::global_ptr<const float> params_ptr(params);

    int r = 0;
    r += process_interaction(iatoms_ptr, params_ptr, tid);
    r += process_interaction(iatoms_ptr, params_ptr, tid);
    result[tid] = r;
  }).wait();

  int first_result, last_result;
  q.memcpy(&first_result, result, sizeof(int));
  q.memcpy(&last_result, result + N-1, sizeof(int));
  q.wait();

  // Each call: params[0]=4, ai=1, aj=2, ak=3 -> 4+1+2+3=10, two calls -> 20
  // CHECK: 20
  std::cout << first_result << std::endl;
  // CHECK: 20
  std::cout << last_result << std::endl;

  sycl::free(iatoms, q);
  sycl::free(params, q);
  sycl::free(result, q);
}
