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



template<class T, int N>
struct vec {
  T data [N];

  vec() {}

  vec(T x) {
    for(int i = 0; i < N; ++i)
      data[i] = static_cast<T>(x);
  }
};

template<class T, int N>
vec<T,N>& operator+=(vec<T,N>& a, const vec<T,N>& b) {
  for(int i=0; i<N; ++i)
    a.data[i] += b.data[i];
  return a;
}

template<class T, int N>
bool operator==(const vec<T,N>& a, const vec<T,N>& b) {
  for(int i = 0; i < N; ++i)
    if(a.data[i] != b.data[i])
      return false;
  return true;
}

template<class T, int N>
bool operator!=(const vec<T,N>& a, const vec<T,N>& b) {
  return !(a==b);
}

template<class T, int N>
std::ostream& operator<<(std::ostream& ostr, const vec<T,N>& a) {
  ostr << "(";
  for(int i = 0; i < N; ++i) {
    ostr << a.data[i];
    if(i != N - 1)
      ostr << ", ";
  }
  ostr << ")";
  return ostr;
}

using test_type = vec<float,4>;

__global__ void reduce_kernel(test_type* in, test_type* out) {
  extern __shared__ test_type local_mem [];
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  local_mem[threadIdx.x] = in[gid];
  __syncthreads();

  if(threadIdx.x == 0) {
    test_type result = local_mem[0];
    for(int i = 1; i < blockDim.x; ++i)
      result += local_mem[i];

    out[blockIdx.x] = result;
  }
}


void run() {
  constexpr std::size_t block_size = 64;
  constexpr std::size_t grid_size = 4;
  std::size_t global_size = grid_size * block_size;

  test_type* in; test_type* out;
  pcudaMallocManaged(&in,  global_size * sizeof(test_type));
  pcudaMallocManaged(&out,  global_size * sizeof(test_type));

  for(int i = 0; i < global_size; ++i)
    in[i] = test_type(i);

  pcudaLaunchKernelGGL(reduce_kernel,
                       grid_size, block_size, block_size * sizeof(test_type), 0, in,
                       out);
  pcudaDeviceSynchronize();

  test_type ref_reduction_results [grid_size];
  for(int g = 0; g < grid_size; ++g) {
    ref_reduction_results[g] = test_type(g*block_size);
    for(int i = 1; i < block_size; ++i) {
      ref_reduction_results[g] += test_type(i+g*block_size);
    }
  }
  
  bool is_reduction_equal = true;
  for(int i = 0; i < grid_size; ++i)
    if(ref_reduction_results[i] != out[i])
      is_reduction_equal = false;
  
  // CHECK: 1
  std::cout << is_reduction_equal << std::endl;

  pcudaFree(in);
  pcudaFree(out);
}

int main() {
  run();
}
