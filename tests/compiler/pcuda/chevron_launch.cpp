// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda --acpp-pcuda-chevron-launch -g
// RUN: %t | FileCheck %s

#include <iostream>
#include <pcuda.hpp>

__global__ void kernel(int* in, int* out) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  out[gid] = in[gid];
}

__global__ void zero_params() {}

int main() {
  std::size_t block_size = 128;
  std::size_t grid_size = 4;
  std::size_t global_size = grid_size * block_size;

  int* in; int* out;

  pcudaMallocManaged(&in,  global_size * sizeof(int));
  pcudaMallocManaged(&out,  global_size * sizeof(int));

  for(int i = 0; i < global_size; ++i)
    in[i] = int{i};

  kernel<<<grid_size, block_size>>>(in, out);

  pcudaDeviceSynchronize();

  bool validates = true;
  for(int i = 0; i < global_size; ++i) {
    if(in[i] != out[i])
      validates = false;
  }

  // CHECK: 1
  if(validates)
    std::cout << "1" << std::endl;
  else
    std::cout << "0" << std::endl;

  if(false)
    // Note: This is not currently expected to run correctly -
    // but it should at least compile
    zero_params<<<1,1>>>();

  pcudaFree(in);
  pcudaFree(out);
}
