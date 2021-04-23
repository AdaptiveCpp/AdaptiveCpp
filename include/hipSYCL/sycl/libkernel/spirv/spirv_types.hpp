//===----------- spirv_types.hpp --- SPIRV types -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdint>

// TODO: include the header file with SPIR-V declarations from SPIRV-Headers
// project.

// Declarations of enums below is aligned with corresponding declarations in
// SPIRV-Headers repo with a few exceptions:
// - base types changed from uint to uint32_t
// - spv namespace renamed to __spv
namespace __spv {

struct Scope {

  enum Flag : uint32_t {
    CrossDevice = 0,
    Device = 1,
    Workgroup = 2,
    Subgroup = 3,
    Invocation = 4,
  };

  constexpr Scope(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};

struct MemorySemanticsMask {

  enum Flag : uint32_t {
    None = 0x0,
    Acquire = 0x2,
    Release = 0x4,
    AcquireRelease = 0x8,
    SequentiallyConsistent = 0x10,
    UniformMemory = 0x40,
    SubgroupMemory = 0x80,
    WorkgroupMemory = 0x100,
    CrossWorkgroupMemory = 0x200,
    AtomicCounterMemory = 0x400,
    ImageMemory = 0x800,
  };

  constexpr MemorySemanticsMask(Flag flag) : flag_value(flag) {}

  constexpr operator uint32_t() const { return flag_value; }

  Flag flag_value;
};

enum class GroupOperation : uint32_t {
  Reduce = 0,
  InclusiveScan = 1,
  ExclusiveScan = 2
};

} // namespace __spv

#ifdef __SYCL_DEVICE_ONLY__
// OpenCL pipe types
template <typename dataT>
using RPipeTy = __attribute__((pipe("read_only"))) const dataT;
template <typename dataT>
using WPipeTy = __attribute__((pipe("write_only"))) const dataT;

// OpenCL vector types
template <typename dataT, int dims>
using __ocl_vec_t = dataT __attribute__((ext_vector_type(dims)));

// Struct representing layout of pipe storage
struct ConstantPipeStorage {
  int32_t _PacketSize;
  int32_t _PacketAlignment;
  int32_t _Capacity;
};

// Arbitrary precision integer type
template <int Bits> using ap_int = _ExtInt(Bits);
#endif // __SYCL_DEVICE_ONLY__

// This class does not have definition, it is only predeclared here.
// The pointers to this class objects can be passed to or returned from
// SPIRV built-in functions.
// Only in such cases the class is recognized as SPIRV type __ocl_event_t.
#ifndef __SYCL_DEVICE_ONLY__
typedef void* __ocl_event_t;
typedef void* __ocl_sampler_t;
// Adding only the datatypes that can be currently used in SYCL,
// as per SYCL spec 1.2.1
#define __SYCL_SPV_IMAGE_TYPE(NAME) typedef void *__ocl_##NAME##_t

#define __SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(NAME)                                \
  __SYCL_SPV_IMAGE_TYPE(NAME);                                                 \
  typedef void *__ocl_sampled_##NAME##_t

__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image1d_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image2d_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image3d_ro);
__SYCL_SPV_IMAGE_TYPE(image1d_wo);
__SYCL_SPV_IMAGE_TYPE(image2d_wo);
__SYCL_SPV_IMAGE_TYPE(image3d_wo);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image1d_array_ro);
__SYCL_SPV_SAMPLED_AND_IMAGE_TYPE(image2d_array_ro);
__SYCL_SPV_IMAGE_TYPE(image1d_array_wo);
__SYCL_SPV_IMAGE_TYPE(image2d_array_wo);

#undef __SYCL_SPV_IMAGE_TYPE
#undef __SYCL_SPV_SAMPLED_AND_IMAGE_TYPE
#endif
