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
#ifndef HIPSYCL_SSCP_BUILTIN_SPIRV_COMMON_HPP
#define HIPSYCL_SSCP_BUILTIN_SPIRV_COMMON_HPP

#include "hipSYCL/sycl/libkernel/sscp/builtins/builtin_config.hpp"



namespace __spv {

enum ScopeFlag : __acpp_uint32 {
  CrossDevice = 0,
  Device = 1,
  Workgroup = 2,
  Subgroup = 3,
  Invocation = 4,
};

enum MemorySemanticsMaskFlag : __acpp_uint32 {
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
  ImageMemory = 0x800
};

}


__attribute__((always_inline)) __spv::ScopeFlag
inline get_spirv_scope(__acpp_sscp_memory_scope scope) {

  if(scope == __acpp_sscp_memory_scope::work_item)
    return __spv::ScopeFlag::Invocation;
  else if(scope == __acpp_sscp_memory_scope::sub_group)
    return __spv::ScopeFlag::Subgroup;
  else if(scope == __acpp_sscp_memory_scope::work_group)
    return __spv::ScopeFlag::Workgroup;
  else if(scope == __acpp_sscp_memory_scope::device)
    return __spv::ScopeFlag::Device;
  else
    return __spv::ScopeFlag::CrossDevice;
}

__attribute__((always_inline)) __spv::MemorySemanticsMaskFlag
inline get_spirv_memory_semantics(__acpp_sscp_memory_order order) {
  if(order == __acpp_sscp_memory_order::seq_cst)
    return __spv::MemorySemanticsMaskFlag::SequentiallyConsistent;
  else if(order == __acpp_sscp_memory_order::acq_rel)
    return __spv::MemorySemanticsMaskFlag::AcquireRelease;
  else if(order == __acpp_sscp_memory_order::release)
    return __spv::MemorySemanticsMaskFlag::Release;
  else if(order == __acpp_sscp_memory_order::acquire)
    return __spv::MemorySemanticsMaskFlag::Acquire;
  else // Relaxed
    return __spv::MemorySemanticsMaskFlag::None;
}


#define __spirv_global __attribute__((address_space(1)))
#define __spirv_local __attribute__((address_space(3)))
#define __spirv_generic

#endif
