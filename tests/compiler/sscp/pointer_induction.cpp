// RUN: %acpp %s -o %t --acpp-targets=generic
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic -O3
// RUN: %t | FileCheck %s

#include "common.hpp"
#include <iostream>
#include <vector>

// Walks a 2D array with pointer induction variables, where the inner loop
// starts at an offset from the pointer of the outer loop.

int main() {
  sycl::queue q = get_queue();

  constexpr int rows = 4;
  constexpr int pitch = 8;
  int *data = sycl::malloc_device<int>(rows * pitch, q);
  int *out = sycl::malloc_device<int>(1, q);

  std::vector<int> init(rows * pitch);
  for (int i = 0; i < rows * pitch; ++i)
    init[i] = i;
  q.copy(init.data(), data, init.size()).wait();

  // Runtime values, so that the loops are not unrolled
  int num_rows = rows, row_pitch = pitch, begin = 1, end = 6;
  q.single_task([=]() {
     int sum = 0;
     int *last_row = data + num_rows * row_pitch;
     for (int *row = data; row < last_row; row += row_pitch)
       for (int *p = row + begin; p < row + end; ++p)
         sum += *p;
     *out = sum;
   }).wait();

  int result = 0;
  q.copy(out, &result, 1).wait();
  // CHECK: 300
  std::cout << result << std::endl;

  sycl::free(data, q);
  sycl::free(out, q);
}
