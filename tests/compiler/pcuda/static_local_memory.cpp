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
__global__ void reduce_kernel(T* in, T* out) {
  __shared__ T local_mem[N];
  int gid = threadIdx.x + blockIdx.x * blockDim.x;

  local_mem[threadIdx.x] = in[gid];
  __syncthreads();

  if(threadIdx.x == 0) {
    T result = local_mem[0];
    for(int i = 1; i < blockDim.x; ++i)
      result += local_mem[i];

    out[blockIdx.x] = result;
  }
}

template<class T1, class T2, class T3, int N>
__global__ void multi_alloc_kernel(T1* out1, T2* out2, T3* out3) {
  __shared__ T1 local_mem1[N];
  __shared__ T2 local_mem2[N];
  __shared__ T3 local_mem3[N];
  
  local_mem1[threadIdx.x] = T1(threadIdx.x);
  local_mem2[threadIdx.x] = T2(threadIdx.x);
  local_mem3[threadIdx.x] = T3(threadIdx.x);
  __syncthreads();
  if(threadIdx.x == 0) {
    out1[blockIdx.x] = local_mem1[N-1];
    out2[blockIdx.x] = local_mem2[N-1];
    out3[blockIdx.x] = local_mem3[N-1];
  }
}

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

template<class T1, class T2, class T3>
void run() {
  constexpr std::size_t block_size = 64;
  constexpr std::size_t grid_size = 4;
  std::size_t global_size = grid_size * block_size;

  T1* in; T1* out1; T2* out2; T3* out3;
  pcudaMallocManaged(&in,  global_size * sizeof(T1));
  pcudaMallocManaged(&out1,  global_size * sizeof(T1));
  pcudaMallocManaged(&out2,  global_size * sizeof(T2));
  pcudaMallocManaged(&out3,  global_size * sizeof(T3));

  for(int i = 0; i < global_size; ++i)
    in[i] = T1(i);

  pcudaLaunchKernelGGL(PCUDA_KERNEL_NAME(reduce_kernel<T1, block_size>),
                       grid_size, block_size, 0, 0, in, out1);
  pcudaDeviceSynchronize();

  T1 ref_reduction_results [grid_size];
  for(int g = 0; g < grid_size; ++g) {
    ref_reduction_results[g] = T1(g*block_size);
    for(int i = 1; i < block_size; ++i) {
      ref_reduction_results[g] += T1(i+g*block_size);
    }
  }
  
  bool is_reduction_equal = true;
  for(int i = 0; i < grid_size; ++i)
    if(ref_reduction_results[i] != out1[i])
      is_reduction_equal = false;
  
  std::cout << is_reduction_equal << std::endl;

  pcudaLaunchKernelGGL(
      PCUDA_KERNEL_NAME(multi_alloc_kernel<T1, T2, T3, block_size>), grid_size,
      block_size, 0, 0, out1, out2, out3);

  pcudaDeviceSynchronize();

  std::cout << (out1[0] == T1{block_size-1}) << std::endl;
  std::cout << (out2[0] == T2{block_size-1}) << std::endl;
  std::cout << (out3[0] == T3{block_size-1}) << std::endl;

  pcudaFree(in);
  pcudaFree(out1);
  pcudaFree(out2);
  pcudaFree(out3);
}

int main() {
  // CHECK: 1
  // CHECK: 1
  // CHECK: 1
  // CHECK: 1
  run<vec<int,4>,double,int>();

  // CHECK: 1
  // CHECK: 1
  // CHECK: 1
  // CHECK: 1
  run<char,vec<double,2>,vec<float,4>>();
}
