// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda -O3
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda -O3 -ffast-math
// RUN: %t | FileCheck %s
// RUN: %acpp %s -o %t --acpp-targets=generic --acpp-pcuda -g
// RUN: %t | FileCheck %s

#include <cstring>
#include <iostream>
#include <pcuda.hpp>


template<class T>
__global__ void copy_kernel(T* in, T* out) {
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  out[gid] = in[gid];
}

template<class T>
__global__ void wrapper(T* in, T* out) {
  // kernels can be called in device code from other kernels
  // (without launching a new kernel, so no dynamic parallelism!)
  copy_kernel(in, out);
}


struct complex_struct {
  int a; int b;
  double c; double d;
  float e; bool f;
  void* ptr;

  complex_struct(int x) {
    a = x;
    b = x;
    c = static_cast<double>(x);
    d = static_cast<double>(x+1);
    e = static_cast<float>(c+d);
    f = x > 4;
    ptr = reinterpret_cast<void*>(x);
  }
};

bool operator==(const complex_struct& a, const complex_struct& b){
  return std::memcmp(&a, &b, sizeof(complex_struct)) == 0;
}

bool operator!=(const complex_struct& a, const complex_struct& b){
  return !(a == b);
}

template<class T, class Launcher>
void run(Launcher l) {
  std::size_t block_size = 128;
  std::size_t grid_size = 4;
  std::size_t global_size = grid_size * block_size;

  T* in; T* out;
  pcudaMallocManaged(&in,  global_size * sizeof(T));
  pcudaMallocManaged(&out,  global_size * sizeof(T));

  for(int i = 0; i < global_size; ++i)
    in[i] = T{i};

  l(grid_size, block_size, in, out);

  pcudaDeviceSynchronize();

  bool validates = true;
  for(int i = 0; i < global_size; ++i) {
    if(in[i] != out[i])
      validates = false;
  }

  if(validates)
    std::cout << "1" << std::endl;
  else
    std::cout << "0" << std::endl;

  pcudaFree(in);
  pcudaFree(out);
}

int main() {
  // CHECK: 1
  run<int>([](auto grid, auto block, auto* in, auto* out) {
    pcudaLaunchKernelGGL(wrapper, grid, block, 0, 0, in, out);
  });

  // CHECK: 1
  run<int>([](auto grid, auto block, auto* in, auto* out) {
    pcudaParallelFor(grid, block, [=](){
      wrapper(in, out);
    });
  });

  // CHECK: 1
  run<complex_struct>([](auto grid, auto block, auto* in, auto* out) {
    pcudaLaunchKernelGGL(wrapper, grid, block, 0, 0, in, out);
  });

  // CHECK: 1
  run<complex_struct>([](auto grid, auto block, auto* in, auto* out) {
    pcudaParallelFor(grid, block, [=](){
      wrapper(in, out);
    });
  });
}
