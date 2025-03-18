/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause


#ifndef ACPP_PCUDA_HPP
#define ACPP_PCUDA_HPP

#include "hipSYCL/glue/llvm-sscp/s1_ir_constants.hpp"
#include "hipSYCL/glue/llvm-sscp/hcf_registration.hpp"
#include "hipSYCL/sycl/libkernel/backend.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/core.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/subgroup.hpp"
#include "hipSYCL/sycl/libkernel/sscp/builtins/barrier.hpp"
#include <cstddef>

#include "detail/vec.hpp"
#include "detail/math.hpp"
#include "pcuda_runtime.hpp"

#ifndef __device__
#define __device__ __attribute__((annotate("hipsycl_sscp_outlining")))
#endif

#ifndef __host__
#define __host__
#endif

#define __global__                                                             \
  __attribute__((annotate("hipsycl_sscp_kernel")))                             \
  __attribute__((annotate("hipsycl_sscp_outlining")))                          \
  __attribute__((annotate("acpp_free_kernel")))

// static local memory is marked using the acpp_local_memory annotation.
// Unfortunately, clang does not codegen this annotation for extern variables
// (i.e. dynamically sized local memory, extern __shared__ syntax),
// so we currently use ABI tags for that purpose as a hack :(
#define __shared__                                                             \
  __attribute__((annotate("acpp_local_memory")))                               \
  __attribute__((abi_tag("__acpp_local_memory_tag__")))

#define PCUDA_BUILTIN_CALL(builtin) if(__acpp_sscp_is_device){builtin;}
#define PCUDA_BUILTIN_CALL_RESULT(builtin, fallback)                           \
  (__acpp_sscp_is_device ? (builtin) : (fallback))

// needs -fdeclspec
struct __pcudaThreadIdx {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_id_x(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_id_y(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_id_z(), 0);
  }
};

struct __pcudaBlockIdx {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_group_id_x(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_group_id_y(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_group_id_z(), 0);
  }
};

struct __pcudaBlockDim {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_size_x(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_size_y(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_local_size_z(), 0);
  }
};

struct __pcudaGridDim {
  __declspec(property(get = __fetch_x)) unsigned x;
  __declspec(property(get = __fetch_y)) unsigned y;
  __declspec(property(get = __fetch_z)) unsigned z;

  operator dim3() { return dim3{x, y, z}; }

  static inline __attribute__((always_inline)) unsigned __fetch_x() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_num_groups_x(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_y() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_num_groups_y(), 0);
  }

  static inline __attribute__((always_inline)) unsigned __fetch_z() {
    return PCUDA_BUILTIN_CALL_RESULT(__acpp_sscp_get_num_groups_z(), 0);
  }
};

template<class F>
__global__
void __pcuda_parallel_for(const F& f){
  if(__acpp_sscp_is_device)
    f();
}

extern const __pcudaThreadIdx threadIdx;
extern const __pcudaBlockIdx blockIdx;
extern const __pcudaBlockDim blockDim;
extern const __pcudaGridDim gridDim;


inline int __pcuda_warp_size() {
  return PCUDA_BUILTIN_CALL_RESULT(
      static_cast<int>(__acpp_sscp_get_subgroup_max_size()), 0);
}
#define warpSize __pcuda_warp_size()

inline void __syncthreads() {
  PCUDA_BUILTIN_CALL(__acpp_sscp_work_group_barrier(
      __acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed));
}

inline void __syncwarp() {
  PCUDA_BUILTIN_CALL(__acpp_sscp_sub_group_barrier(
      __acpp_sscp_memory_scope::work_group, __acpp_sscp_memory_order::relaxed));
}

///////////////// Kernel launch mechanisms ////////////////////

struct __pcuda_dummy_configuration_t {};

template<class T>
inline T operator*(__pcuda_dummy_configuration_t, T x) {
  return x;
}
__pcuda_dummy_configuration_t inline __pcuda_pp_generate_configuration(
    dim3 grid, dim3 block, size_t shared_mem = 0, pcudaStream_t s = nullptr) {
  __pcudaPushCallConfiguration(grid, block, shared_mem, s);
  return __pcuda_dummy_configuration_t{};
}

template <class F>
inline pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t shared_mem,
                                pcudaStream_t stream, F f) {
  __pcudaPushCallConfiguration(grid, block, shared_mem, stream);
  // Compiler looks for this alloca
  F g = f;
  __pcuda_parallel_for(g);
  return pcudaGetLastError();
}

template<class F>
inline pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, size_t shared_mem, F f) {
  return pcudaParallelFor(grid, block, 0, nullptr, f);
}

template<class F>
inline pcudaError_t pcudaParallelFor(dim3 grid, dim3 block, F f) {
  return pcudaParallelFor(grid, block, 0, f);
}

#define PCUDA_KERNEL_NAME(...) __VA_ARGS__
#define PCUDA_SYMBOL(X) X
#define pcudaLaunchKernelGGL(kernel_name, grid, block, shared_mem, stream,     \
                             ...)                                              \
  (__pcudaPushCallConfiguration(grid, block, shared_mem, stream),              \
   kernel_name(__VA_ARGS__))

#endif
