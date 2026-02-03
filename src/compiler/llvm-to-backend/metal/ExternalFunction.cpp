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
#include "ExternalFunction.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace hipsycl {
namespace compiler {

namespace {

std::vector<ExternalFunctionInfo> generateIdFunctions(const std::string& baseName, const std::string& attribute) {
  return {
    {
      .name = baseName,
      .code = "uint3 " + baseName + " [[ " + attribute + " ]];\n",
    },
    {
      .name = baseName + "_x",
      .code = "inline ulong " + baseName + "_x() { return " + baseName + ".x; }\n",
      .deps = {baseName},
    },
    {
      .name = baseName + "_y",
      .code = "inline ulong " + baseName + "_y() { return " + baseName + ".y; }\n",
      .deps = {baseName},
    },
    {
      .name = baseName + "_z",
      .code = "inline ulong " + baseName + "_z() { return " + baseName + ".z; }\n",
      .deps = {baseName},
    },
  };
}

std::vector<ExternalFunctionInfo> generateHalfOps() {
  const std::vector<std::pair<std::string, std::string>> ops = {
    {"add", "+"},
    {"sub", "-"},
    {"mul", "*"},
    {"div", "/"},
  };

  std::vector<ExternalFunctionInfo> functions;
  for (const auto& [name, symbol] : ops) {
    std::string code = "inline ushort __acpp_sscp_half_" + name + "(ushort a, ushort b) { ";
    code += " return as_type<ushort>(as_type<half>(a) " + symbol + " as_type<half>(b)); }\n";
    functions.push_back(ExternalFunctionInfo{
      .name = "__acpp_sscp_half_" + name,
      .code = code,
    });
  }
  return functions;
}

std::vector<ExternalFunctionInfo> generateSimpleMathFunctions() {
  // warning: atan2(0, 0) is nan in metal, but in C:
  // atan2(±0, −0) returns ±π
  // atan2(±0, +0) returns ±0.
  const std::vector<std::string> funcNames = {
    "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "cos", "sin",
    "exp", "exp2", "exp10",
    "log", "log2", "log10",
    "sqrt", "rsqrt",
    "floor", "ceil", "round", "trunc", "rint",
    "fabs", "ldexp",
    "ctz", "clz", "popcount",
    "copysign", "fma", "isnan", "isinf",
    "isfinite", "isnormal", "signbit",
    "fmin", "fmax", "fmod", "fdim",
  };

  std::vector<ExternalFunctionInfo> result;
  for (const auto& func : funcNames) {
    result.push_back({
      .name = "__acpp_sscp_" + func,
      .replacement = func,
      .exactMatch = false
    });
  }

  result.push_back({
    .name = "__acpp_sscp_hypot_f32",
    .code = R"__(
  inline float __acpp_sscp_hypot_f32(float x, float y) {
    return length(float2(x, y));
  }
)__",
  });
  result.push_back({
    .name = "__acpp_sscp_mad",
    .replacement = "fma",
    .exactMatch = false,
  });
  result.push_back({
    .name = "__acpp_sscp_rootn_f32",
    .code = R"__(
inline float __acpp_sscp_rootn_f32(float x, int n) {
  if (n == 0) return NAN;

  if (x < 0.0f) {
    if ((n & 1) == 0)
        return NAN;
    return -pow(-x, 1.0f / float(n));
  }

  return pow(x, 1.0f / float(n));
}
)__",
  });
  result.push_back({
    .name = "__acpp_sscp_expm1_f32",
    .code = R"__(
inline float __acpp_sscp_expm1_f32(float x) {
  float u = exp(x);
  if (u == 1.0f) return x;
  if (u - 1.0f == -1.0f) return -1.0f;
  return (u - 1.0f) * x / log(u);
}
)__",
  });
  result.push_back({
    .name = "__acpp_sscp_log1p_f32",
    .code = R"__(
inline float __acpp_sscp_log1p_f32(float x) {
  float u = 1.0f + x;
  if (u == 1.0f) return x;
  return log(u) * x / (u - 1.0f);
}
)__",
  });
  return result;
}

std::vector<ExternalFunctionInfo> generateLLVMIntrinsics() {
  const std::vector<std::tuple<std::string, std::string, int>> intrinsics = {
    {"ctlz", "clz", 1},
    {"cttz", "ctz", 1},
    {"ctpop", "popcount", -1},
    {"umin", "min", 2},
    {"umax", "max", 2},
    {"maxnum", "fmax", 2},
    {"minnum", "fmin", 2},
    {"atan2", "atan2", 2},
    {"atan", "atan", 1},
    {"copysign", "copysign", 2},
    {"floor", "floor", 1},
    {"fabs", "fabs", 1},
    {"fmuladd", "fma", 3},
  };

  std::vector<ExternalFunctionInfo> result;
  for (const auto& [llvmName, metalName, argCount] : intrinsics) {
    result.push_back({
      .name = "llvm." + llvmName + ".",
      .replacement = metalName,
      .exactMatch = false,
      .argsCount = argCount
    });
  }
  return result;
}

std::vector<ExternalFunctionInfo> generateIgnorableIntrinsics() {
  const std::vector<const char*> ignorable = {
    "llvm.lifetime.",
    "llvm.dbg.",
    "llvm.assume",
    "llvm.invariant.",
    "llvm.experimental.noalias.scope.decl"
  };

  std::vector<ExternalFunctionInfo> result;
  for (const auto& name : ignorable) {
    result.push_back({
      .name = name,
      .exactMatch = false,
      .ignore = true
    });
  }
  return result;
}

std::vector<ExternalFunctionInfo> initExternalFunctionTable() {
  std::vector<ExternalFunctionInfo> externalFunctions = {
    // memcpy
    {
      .name = "llvm.memcpy",
      .replacement = "memcpy",
      .code = R"__(
  #define __MEMCPY(addrspace1, addrspace2) \
  inline void memcpy(addrspace1 void* dst, const addrspace2 void* src, size_t size) { \
      for (size_t i = 0; i < size; ++i) { \
          ((addrspace1 uchar*)dst)[i] = ((const addrspace2 uchar*)src)[i]; \
      } \
  }

  #define __MEMCPY_FOR_EACH_SRC(M, DST_AS) \
      M(DST_AS, thread) \
      M(DST_AS, threadgroup) \
      M(DST_AS, device) \
      M(DST_AS, constant)

  #define __MEMCPY_FOR_EACH_DST(M) \
      __MEMCPY_FOR_EACH_SRC(M, thread) \
      __MEMCPY_FOR_EACH_SRC(M, threadgroup) \
      __MEMCPY_FOR_EACH_SRC(M, device)

  __MEMCPY_FOR_EACH_DST(__MEMCPY)
      )__",
      .exactMatch = false, // exactMatch
      .argsCount = 3
    },
    // end of memcpy

    // memset
    {
      .name = "llvm.memset",
      .replacement = "memset",
      .code = R"__(
  #define __MEMSET(addrspace) \
  inline void memset(addrspace void* dst, int value, size_t size) { \
    for (size_t i = 0; i < size; ++i) { \
      ((addrspace uchar*)dst)[i] = (uchar)value; \
    } \
  }

  #define __MEMSET_FOR_EACH_DST \
      __MEMSET(thread) \
      __MEMSET(threadgroup) \
      __MEMSET(device)

  __MEMSET_FOR_EACH_DST
  )__",
      .exactMatch = false,
      .argsCount = 3
    },
    // end of memset

    // subgroups
    {
      .name = "__acpp_sscp_get_subgroup_max_size",
      .code = R"__(
  uint __simd_size [[threads_per_simdgroup]];
  uint __simd_group_id [[simdgroup_index_in_threadgroup]];
  inline uint32_t __acpp_sscp_get_subgroup_max_size() {
    return __simd_size;
  }
  )__",
    },
    {
      .name = "__acpp_sscp_get_subgroup_size",
      .code = R"__(
  inline uint32_t __acpp_sscp_get_subgroup_size() {
    const uint sg = __simd_size;

    const uint3 lid3 = __acpp_sscp_get_local_id;
    const uint3 lsz3 = __acpp_sscp_get_local_size;

    const uint lx = lsz3.x;
    const uint ly = lsz3.y;

    const uint lid = lid3.x + lid3.y * lx + lid3.z * (lx * ly);
    const uint wg  = lsz3.x * lsz3.y * lsz3.z;

    const uint start = (lid / sg) * sg;
    const uint rem = wg - start;

    return (rem < sg) ? rem : sg;
  }
  )__",
      .deps = {"__acpp_sscp_get_subgroup_max_size", "__acpp_sscp_get_local_id", "__acpp_sscp_get_local_size"}
    },
    {
      .name = "__acpp_sscp_get_subgroup_local_id",
      .code = R"__(
  uint __simd_lane_id [[thread_index_in_simdgroup]];
  inline uint32_t __acpp_sscp_get_subgroup_local_id() {
    return __simd_lane_id;
  }
  )__",
    },

    {
      .name = "__acpp_sscp_get_dynamic_local_memory",
      .convertToVar = true
    },
    {
      "__acpp_sscp_work_group_barrier",
      std::nullopt,
  R"__(
  inline void __acpp_sscp_work_group_barrier(uint32_t mem_scope, uint32_t mem_order) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (mem_order == 0) {
        threadgroup_barrier(mem_flags::mem_none);
      } else {
        if (mem_scope >= 3) {
          threadgroup_barrier(mem_flags::mem_device);
        } else {
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }
    }
  )__",
      true,
    },

    // overflow
    {
      "llvm.uadd.with.overflow.",
      "__acpp_sscp_uadd_with_overflow",
R"__(
  template<typename R, typename T>
  inline R __acpp_sscp_uadd_with_overflow(T a, T b) {
      R result;
      result.field0 = a + b;
      result.field1 = (result.field0 < a);
      return result;
  }
)__",
      false, // exactMatch
    },
    {
      "llvm.usub.with.overflow.",
      "__acpp_sscp_usub_with_overflow",
R"__(
  template<typename R, typename T>
  inline R __acpp_sscp_usub_with_overflow(T a, T b) {
      R result;
      result.field0 = a - b;
      result.field1 = (a < b);
      return result;
  }
)__",
      false, // exactMatch
    },

    {
      "llvm.fshl.",
      "__acpp_sscp_fshl",
R"__(
  template<typename T>
  inline T __acpp_sscp_fshl(T a, T b, uint32_t shift) {
      const uint32_t bitWidth = sizeof(T) * 8;
      shift = shift % bitWidth;
      if (shift == 0) {
          return a;
      }
      return (a << shift) | (b >> (bitWidth - shift));
  };
)__",
      false, // exactMatch
    },

    // fshr
    {
      "llvm.fshr.",
      "__acpp_sscp_fshr",
R"__(
  template<typename T>
  inline T __acpp_sscp_fshr(T a, T b, uint32_t shift) {
      const uint32_t bitWidth = sizeof(T) * 8;
      shift = shift % bitWidth;
      if (shift == 0) {
          return a;
      }
      return (a >> shift) | (b << (bitWidth - shift));
  };
)__",
      false, // exactMatch
    },

    {
      .name = "__as_signed",
      .code = R"__(
  inline char __as_signed(uchar value) {
    return as_type<char>(value);
  }

  inline short __as_signed(ushort value) {
    return as_type<short>(value);
  }

  inline int __as_signed(uint value) {
    return as_type<int>(value);
  }

  inline long __as_signed(ulong value) {
    return as_type<long>(value);
  }

  inline bool __as_signed(bool value) {
    return value;
  }
)__",
      .exactMatch = false,
      .used = true // mark as used to always include
    },
    // smax <- signed max
    {
      "llvm.smax.",
      "__acpp_sscp_smax",
R"__(
  template<typename T1, typename T2>
  inline auto __acpp_sscp_smax(T1 a, T2 b) {
      auto sa = __as_signed(a);
      auto sb = __as_signed(b);
      return (sa > sb) ? a : b;
  }
)__",
      false, // exactMatch
      false,
      {"__as_signed"}
    },
    {
      "llvm.smin.",
      "__acpp_sscp_smin",
R"__(
  template<typename T1, typename T2>
  inline auto __acpp_sscp_smin(T1 a, T2 b) {
      auto sa = __as_signed(a);
      auto sb = __as_signed(b);
      return (sa < sb) ? a : b;
    }
)__",
      false, // exactMatch
      false,
      {"__as_signed"}
    },
    {
      .name = "llvm.abs.",
      .replacement = "__acpp_sscp_abs",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_abs(T value) {
  auto svalue = __as_signed(value);
  decltype(svalue) sresult = (svalue < 0) ? -svalue : svalue;
  return as_type<T>(sresult);
}
)__",
      .exactMatch = false,
      .deps = {"__as_signed"},
      .argsCount = 1,
    },

    {
      .name = "__acpp_sscp_pow_f32",
      .code = R"__(
inline float __acpp_sscp_pow_f32(float base, float exp) {
  return pow(base, exp);
})__",
    },
    {
      .name = "__acpp_sscp_powr_f32",
      .code = R"__(
inline float __acpp_sscp_powr_f32(float base, float exp) {
  return pow(base, exp);
})__",
    },
    {
      .name = "__acpp_sscp_pown_f32",
      .code = R"__(
inline float __acpp_sscp_pown_f32(float base, int exp) {
  return pow(base, exp);
})__",
    },
    {
      .name = "__acpp_sscp_work_group_inclusive",
      .code = R"__(
template<int op, typename T>
inline T __acpp_sscp_work_group_inclusive(T value, threadgroup T* scratch, uint32_t local_mem_size) {
  const uint subgroup_size = __simd_size;
  const uint3 l = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  scratch = (threadgroup T*)((threadgroup uchar*)(scratch) + local_mem_size);

  const uint lid = (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  const uint local_size = (uint)tg.x * (uint)tg.y * (uint)tg.z;

  const uint group_id = __simd_group_id;
  const uint lane_id  = __simd_lane_id;
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  auto prefix_op = [&](T value) {
    if constexpr(op == 0) {
      return simd_prefix_inclusive_sum(value);
    } else {
      return simd_prefix_inclusive_product(value);
    }
  };

  auto binary_op = [&](T a, T b) {
    if constexpr(op == 0) {
      return a + b;
    } else {
      return a * b;
    }
  };

  auto initial_value = [&]() {
    if constexpr(op == 0) {
      return T{0};
    } else {
      return T{1};
    }
  };

  const T prefix = prefix_op(value);

  const uint group_base = group_id * subgroup_size;
  uint active = 0;
  if(group_base < local_size) {
    uint rem = local_size - group_base;
    active = (rem < subgroup_size) ? rem : subgroup_size;
  }
  const uint last_lane = (active > 0) ? (active - 1u) : 0u;

  if(lane_id == last_lane) {
    scratch[group_id] = prefix;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(ngroups <= subgroup_size) {
    if(lid < ngroups) {
      T v = scratch[lid];
      T p = prefix_op(v);
      scratch[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if(ngroups <= 2u * subgroup_size) {
    if(lid < ngroups) {
      T v = scratch[lid];
      T p = prefix_op(v);
      scratch[lid] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(lid < ngroups) {
      T add = (lid >= subgroup_size) ? scratch[subgroup_size - 1u] : initial_value();
      scratch[lid] = binary_op(scratch[lid], add);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    for(uint offset = 1; offset < ngroups; offset <<= 1) {
      T addend = initial_value();
      if(lid < ngroups && lid >= offset) {
        addend = scratch[lid - offset];
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if(lid < ngroups && lid >= offset) {
        scratch[lid] = binary_op(scratch[lid], addend);
      }

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const T group_offset = (group_id > 0) ? scratch[group_id - 1] : initial_value();
  return binary_op(prefix, group_offset);
}
)__",
      .deps = {
        "__acpp_sscp_get_group_id",
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_num_groups",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_subgroup_max_size",
        "__acpp_sscp_get_subgroup_local_id"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_work_group_exclusive",
      .code = R"__(
template<int op, typename T>
inline T __acpp_sscp_work_group_exclusive(T value, threadgroup T* scratch, uint32_t local_mem_size) {
  const uint subgroup_size = __simd_size;
  const uint3 l  = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  scratch = (threadgroup T*)((threadgroup uchar*)(scratch) + local_mem_size);

  const uint lid = (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  const uint local_size = (uint)tg.x * (uint)tg.y * (uint)tg.z;

  const uint group_id = __simd_group_id;
  const uint lane_id  = __simd_lane_id;
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  auto prefix_incl_op = [&](T x) {
    if constexpr(op == 0) return simd_prefix_inclusive_sum(x);
    else return simd_prefix_inclusive_product(x);
  };

  auto binary_op = [&](T a, T b) {
    if constexpr(op == 0) return a + b;
    else return a * b;
  };

  auto identity = [&]() {
    if constexpr(op == 0) return T{0};
    else return T{1};
  };

  const uint group_base = group_id * subgroup_size;
  uint active = 0;
  if(group_base < local_size) {
    uint rem = local_size - group_base;
    active = (rem < subgroup_size) ? rem : subgroup_size;
  }
  const uint last_lane = (active > 0) ? (active - 1u) : 0u;

  T v = value;
  if(lane_id >= active) v = identity();

  const T incl = prefix_incl_op(v);

  T excl = simd_shuffle_up(incl, 1); // lane i gets incl(i-1)
  if(lane_id == 0) excl = identity();

  if(lane_id == last_lane) scratch[group_id] = incl;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(ngroups <= subgroup_size) {
    if(lid < ngroups) {
      T s = scratch[lid];
      scratch[lid] = prefix_incl_op(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else if(ngroups <= 2u * subgroup_size) {
    if(lid < ngroups) {
      T s = scratch[lid];
      scratch[lid] = prefix_incl_op(s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if(lid < ngroups) {
      T add = (lid >= subgroup_size) ? scratch[subgroup_size - 1u] : identity();
      scratch[lid] = binary_op(scratch[lid], add);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  } else {
    for(uint offset = 1; offset < ngroups; offset <<= 1) {
      T addend = identity();
      if(lid < ngroups && lid >= offset) addend = scratch[lid - offset];

      threadgroup_barrier(mem_flags::mem_threadgroup);

      if(lid < ngroups && lid >= offset) scratch[lid] = binary_op(scratch[lid], addend);

      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const T group_offset = (group_id > 0) ? scratch[group_id - 1] : identity();
  return binary_op(excl, group_offset);
}
)__",
      .deps = {
        "__acpp_sscp_get_group_id",
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_num_groups",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_subgroup_max_size",
        "__acpp_sscp_get_subgroup_local_id"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_work_group_reduce_helper",
      .code = R"__(
template<int op, typename T>
inline T __acpp_sscp_work_group_reduce_helper(T value, threadgroup T* scratch, uint32_t local_mem_size) {
  const uint subgroup_size = __simd_size;
  const uint3 l  = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  scratch = (threadgroup T*)((threadgroup uchar*)(scratch) + local_mem_size);

  const uint lid =
      (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  const uint local_size = (uint)tg.x * (uint)tg.y * (uint)tg.z;

  const uint group_id = __simd_group_id;
  const uint lane_id  = __simd_lane_id;
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  auto reduce_op = [&](T v) {
    if constexpr(op == 0) return simd_sum(v);
    else return simd_product(v);
  };

  auto binary_op = [&](T a, T b) {
    if constexpr(op == 0) return a + b;
    else return a * b;
  };

  auto identity = [&]() {
    if constexpr(op == 0) return T{0};
    else return T{1};
  };

  const uint group_base = group_id * subgroup_size;
  uint active = 0;
  if(group_base < local_size) {
    uint rem = local_size - group_base;
    active = (rem < subgroup_size) ? rem : subgroup_size;
  }

  T v = value;
  if(lane_id >= active) v = identity();

  const T sg_reduced = reduce_op(v);

  if(lane_id == 0) scratch[group_id] = sg_reduced;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if(group_id == 0) {
    T result = identity();

    if(ngroups <= subgroup_size) {
      T x = identity();
      if(lid < ngroups) x = scratch[lid];
      result = reduce_op(x);

      if(lane_id == 0) scratch[0] = result;
    }
    else if(ngroups <= 2u * subgroup_size) {
      T x0 = identity();
      if(lid < subgroup_size) {
        uint i = lid;
        if(i < ngroups) x0 = scratch[i];
      }
      T r0 = reduce_op(x0);

      T x1 = identity();
      if(lid < subgroup_size) {
        uint i = lid + subgroup_size;
        if(i < ngroups) x1 = scratch[i];
      }
      T r1 = reduce_op(x1);

      if(lane_id == 0) scratch[0] = r0;
      if(lane_id == 1) scratch[1] = r1;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      T y = identity();
      if(lid < 2) y = scratch[lid];
      result = reduce_op(y);

      if(lane_id == 0) scratch[0] = result;
    }
    else {
      uint active_ngroups = ngroups;
      while(active_ngroups > 1) {
        uint offset = (active_ngroups + 1u) >> 1;
        if(lid < offset) {
          uint j = lid + offset;
          if(j < active_ngroups)
            scratch[lid] = binary_op(scratch[lid], scratch[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        active_ngroups = offset;
      }
      if(lane_id == 0) scratch[0] = scratch[0];
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  return scratch[0];
}
)__",
      .deps = {
        "__acpp_sscp_get_group_id",
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_num_groups",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_subgroup_max_size",
        "__acpp_sscp_get_subgroup_local_id"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_work_group_broadcast",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_work_group_broadcast(uint32_t local_id, T value, threadgroup void* scratch, uint32_t local_mem_size) {
  const uint3 l = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  const uint lid = (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  scratch = ((threadgroup uchar*)(scratch) + local_mem_size);

  if (lid == local_id) {
    ((threadgroup T*)scratch)[0] = value;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return ((threadgroup T*)scratch)[0];
}
)__",
      .exactMatch = false,
      .deps = {
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_local_size"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_sub_group_broadcast",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_broadcast(uint32_t local_id, T value) {
  if constexpr(sizeof(T) <= 4) {
    return simd_broadcast(value, local_id);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_broadcast(in.parts[i], local_id);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_shl",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_shl(T value, uint32_t shift) {
  if constexpr(sizeof(T) <= 4) {
    return simd_shuffle_down(value, shift);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_shuffle_down(in.parts[i], shift);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_shr",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_shr(T value, uint32_t shift) {
  if constexpr(sizeof(T) <= 4) {
    return simd_shuffle_up(value, shift);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_shuffle_up(in.parts[i], shift);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_permute",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_permute(T value, uint32_t mask) {
  if constexpr(sizeof(T) <= 4) {
    return simd_shuffle_xor(value, mask);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_shuffle_xor(in.parts[i], mask);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_permute",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_permute(T value, uint32_t mask) {
  if constexpr(sizeof(T) <= 4) {
    return simd_shuffle_xor(value, mask);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_shuffle_xor(in.parts[i], mask);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_select",
      .code = R"__(
template<typename T>
inline T __acpp_sscp_sub_group_select(T value, uint32_t lane) {
  if constexpr(sizeof(T) <= 4) {
    return simd_shuffle(value, lane);
  } else {
    union {
      T value;
      uint32_t parts[sizeof(T) / 4];
    } in, out;
    in.value = value;
    for (size_t i = 0; i < sizeof(T) / 4; ++i) {
      out.parts[i] = simd_shuffle(in.parts[i], lane);
    }
    return out.value;
  }
}
)__",
      .exactMatch = false,
    },
    {
      .name = "__acpp_sscp_sub_group_any",
      .replacement = "simd_any",
    },
    {
      .name = "__acpp_sscp_sub_group_all",
      .replacement = "simd_all",
    },
    {
      .name = "__acpp_sscp_sub_group_none",
      .code = R"__(
inline bool __acpp_sscp_sub_group_none(bool predicate) {
  return !simd_any(predicate);
}
)__",
    },
    {
      .name = "__acpp_sscp_work_group_any",
      .code = R"__(
inline bool __acpp_sscp_work_group_any(bool predicate, threadgroup void* scratch, uint32_t local_mem_size) {
  const uint3 l = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  const uint lid = (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  const uint local_size = (uint)tg.x * (uint)tg.y * (uint)tg.z;
  scratch = ((threadgroup uchar*)(scratch) + local_mem_size);

  const uint group_id = __simd_group_id;
  const uint lane_id  = __simd_lane_id;
  const uint subgroup_size = __simd_size;
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  bool sg_any = simd_any(predicate);
  if (lane_id == 0) {
    ((threadgroup bool*)scratch)[group_id] = sg_any;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (group_id == 0) {
    bool v = false;
    if (lid < ngroups) {
      v = ((threadgroup bool*)scratch)[lid];
    }
    bool wg_any = simd_any(v);
    if (lid == 0) {
      ((threadgroup bool*)scratch)[0] = wg_any;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  return ((threadgroup bool*)scratch)[0];
}
)__",
      .deps = {
        "__acpp_sscp_get_group_id",
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_num_groups",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_subgroup_max_size",
        "__acpp_sscp_get_subgroup_local_id"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_work_group_all",
      .code = R"__(
inline bool __acpp_sscp_work_group_all(bool predicate, threadgroup void* scratch, uint32_t local_mem_size) {
  const uint3 l = __acpp_sscp_get_local_id;
  const uint3 tg = __acpp_sscp_get_local_size;
  const uint lid = (uint)l.x + (uint)tg.x * ((uint)l.y + (uint)tg.y * (uint)l.z);
  const uint local_size = (uint)tg.x * (uint)tg.y * (uint)tg.z;
  scratch = ((threadgroup uchar*)(scratch) + local_mem_size);

  const uint group_id = __simd_group_id;
  const uint lane_id  = __simd_lane_id;
  const uint subgroup_size = __simd_size;
  const uint ngroups  = (local_size + subgroup_size - 1u) / subgroup_size;

  bool sg_all = simd_all(predicate);
  if (lane_id == 0) {
    ((threadgroup bool*)scratch)[group_id] = sg_all;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (group_id == 0) {
    bool v = true;
    if (lid < ngroups) {
      v = ((threadgroup bool*)scratch)[lid];
    }
    bool wg_all = simd_all(v);
    if (lid == 0) {
      ((threadgroup bool*)scratch)[0] = wg_all;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  return ((threadgroup bool*)scratch)[0];
}
)__",
      .deps = {
        "__acpp_sscp_get_group_id",
        "__acpp_sscp_get_local_id",
        "__acpp_sscp_get_num_groups",
        "__acpp_sscp_get_subgroup_size",
        "__acpp_sscp_get_subgroup_max_size",
        "__acpp_sscp_get_subgroup_local_id",
        "__acpp_sscp_get_local_size"
      },
      .needsLocalMemory = true,
    },
    {
      .name = "__acpp_sscp_work_group_none",
      .code = R"__(
inline bool __acpp_sscp_work_group_none(bool predicate, threadgroup void* scratch, uint32_t local_mem_size) {
  return !__acpp_sscp_work_group_any(predicate, scratch, local_mem_size);
}
)__",
      .deps = {
        "__acpp_sscp_work_group_any",
      },
      .needsLocalMemory = true,
    },
    {
      .name = "i48u",
      .code = R"__(
struct i48u {
  packed_ushort3 w;
  i48u() : w(packed_ushort3(0,0,0)) {}
  explicit i48u(packed_ushort3 ww) : w(ww) {}
  explicit i48u(ushort x) : w(packed_ushort3(x, 0, 0)) {}
  explicit i48u(uint x)
  : w(packed_ushort3((ushort)(x & 0xffffu),
                     (ushort)((x >> 16) & 0xffffu),
                     0))
  {}
  explicit i48u(ulong x)
  : w(packed_ushort3((ushort)(x & 0xfffful),
                     (ushort)((x >> 16) & 0xfffful),
                     (ushort)((x >> 32) & 0xfffful)))
  {}

  friend inline i48u operator|(i48u a, i48u b) {
    return i48u(packed_ushort3((ushort)(a.w[0] | b.w[0]),
                               (ushort)(a.w[1] | b.w[1]),
                               (ushort)(a.w[2] | b.w[2])));
  }

  friend inline i48u operator<<(i48u a, uint bits) {
    uint s = bits >> 4; // /16
    if ((bits & 0xFu) != 0) {
      ulong x = a.to_ulong();
      x = (x << bits) & 0x0000FFFFFFFFFFFFul;
      return i48u(x);
    }
    if (s == 0) return a;
    if (s == 1) return i48u(packed_ushort3(0, a.w[0], a.w[1]));
    if (s == 2) return i48u(packed_ushort3(0, 0, a.w[0]));
    return i48u(); // >=48 => 0
  }

  friend inline i48u operator>>(i48u a, uint bits) {
    uint s = bits >> 4; // /16
    if ((bits & 0xFu) != 0) {
      ulong x = a.to_ulong();
      x = (x >> bits);
      return i48u(x);
    }
    if (s == 0) return a;
    if (s == 1) return i48u(packed_ushort3(a.w[1], a.w[2], 0));
    if (s == 2) return i48u(packed_ushort3(a.w[2], 0, 0));
    return i48u(); // >=48 => 0
  }

  inline ulong to_ulong() const {
    return (ulong)w[0] | ((ulong)w[1] << 16) | ((ulong)w[2] << 32);
  }
};
)__",
    },
    {
      .name = "atomic_fetch_max_min_float",
      .code = R"__(
inline float __acpp_atomic_fetch_min(device atomic_float* addr_bits, float operand)
{
  device void* p = addr_bits;
  device atomic_uint* addr = (device atomic_uint*)(p);
  uint old_bits = atomic_load_explicit(addr, memory_order_relaxed);
  while (true) {
    float old_val = as_type<float>(old_bits);
    float new_val = fmin(old_val, operand);
    uint  new_bits = as_type<uint>(new_val);

    // If no change needed, return old
    if (new_bits == old_bits)
      return old_val;

    uint expected = old_bits;
    bool ok = atomic_compare_exchange_weak_explicit(
        addr, &expected, new_bits,
        memory_order_relaxed, memory_order_relaxed);

    if (ok)
      return old_val;

    old_bits = expected;
  }
}

inline float __acpp_atomic_fetch_max(device atomic_float* addr_bits, float operand)
{
  device void* p = addr_bits;
  device atomic_uint* addr = (device atomic_uint*)(p);
  uint old_bits = atomic_load_explicit(addr, memory_order_relaxed);
  while (true) {
    float old_val = as_type<float>(old_bits);
    float new_val = fmax(old_val, operand);
    uint  new_bits = as_type<uint>(new_val);

    // If no change needed, return old
    if (new_bits == old_bits)
      return old_val;

    uint expected = old_bits;
    bool ok = atomic_compare_exchange_weak_explicit(
        addr, &expected, new_bits,
        memory_order_relaxed, memory_order_relaxed);

    if (ok)
      return old_val;

    old_bits = expected;
  }
}
)__",
    }
  };

  auto append = [&](auto&& funcs) {
    externalFunctions.insert(
      externalFunctions.end(),
      std::make_move_iterator(funcs.begin()),
      std::make_move_iterator(funcs.end())
    );
  };

  append(generateIdFunctions("__acpp_sscp_get_group_id", "threadgroup_position_in_grid"));
  append(generateIdFunctions("__acpp_sscp_get_num_groups", "threadgroups_per_grid"));
  append(generateIdFunctions("__acpp_sscp_get_local_id", "thread_position_in_threadgroup"));
  append(generateIdFunctions("__acpp_sscp_get_local_size", "threads_per_threadgroup"));
  append(generateHalfOps());
  append(generateSimpleMathFunctions());
  append(generateLLVMIntrinsics());
  append(generateIgnorableIntrinsics());

  return externalFunctions;
}

} // namespace

ExternalFunctionMapper::ExternalFunctionMapper(std::function<std::string(const llvm::Value*)> addrSpaceMapper, std::function<std::string(const llvm::Value*)> exprMapper, std::function<std::string(const llvm::Type*)> typeMapper)
    : externalFunctions(initExternalFunctionTable()),
      addrSpaceMapper(addrSpaceMapper),
      exprMapper(exprMapper),
      typeMapper(typeMapper)
{
  initializeMap();
}

const ExternalFunctionInfo* ExternalFunctionMapper::getFunctionInfo(std::string_view name) {
  auto it = map.upper_bound(name);
  while (it != map.begin()) {
    --it;
    if (name.find(it->first) == 0) {
      ExternalFunctionInfo* info = it->second;
      if (info->exactMatch && info->name != name) {
          return nullptr;
      }
      info->used = true;
      return info;
    }
  }
  return nullptr;
}

std::vector<const ExternalFunctionInfo*> ExternalFunctionMapper::getUsedFunctions() {
  std::vector<const ExternalFunctionInfo*> used;
  std::unordered_set<std::string> added;

  std::function<void(const ExternalFunctionInfo&)> addWithDeps = [&](const ExternalFunctionInfo& funcInfo) {
    for (const auto& depName : funcInfo.deps) {
      auto depInfo = getFunctionInfo(depName);
      if (depInfo) {
        addWithDeps(*depInfo);
      }
    }
    if (added.count(funcInfo.name) > 0) {
      return;
    }
    added.insert(funcInfo.name);
    used.push_back(&funcInfo);
  };

  for (const auto& funcInfo : externalFunctions) {
    if (funcInfo.used) {
      addWithDeps(funcInfo);
    }
  }
  return used;
}

void ExternalFunctionMapper::initializeMap() {
  ExternalFunctionInfo atomicInfo = {
    .name = "__acpp_sscp_atomic",
    .exactMatch = false,
    .deps = {
      "atomic_fetch_max_min_float",
    },
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitAtomicCall(CI, errorStr);
    },
  };
  externalFunctions.push_back(atomicInfo);

  ExternalFunctionInfo memoryFenceInfo = {
    .name = "__acpp_sscp_memory_fence",
    .exactMatch = true,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitMemoryFenceCall(CI, errorStr);
    }
  };
  externalFunctions.push_back(memoryFenceInfo);

  ExternalFunctionInfo subgroupScanInfo = {
    .name = "__acpp_sscp_sub_group_inclusive_scan",
    .exactMatch = false,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitSubgroupScanCall(CI, errorStr);
    }
  };
  externalFunctions.push_back(subgroupScanInfo);

  ExternalFunctionInfo subgroupExclusiveScanInfo = {
    .name = "__acpp_sscp_sub_group_exclusive_scan",
    .exactMatch = false,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitSubgroupExclusiveScanCall(CI, errorStr);
    }
  };
  externalFunctions.push_back(subgroupExclusiveScanInfo);

  ExternalFunctionInfo subgroupReduceInfo = {
    .name = "__acpp_sscp_sub_group_reduce",
    .exactMatch = false,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitSubgroupReduceCall(CI, errorStr);
    }
  };
  externalFunctions.push_back(subgroupReduceInfo);

  ExternalFunctionInfo subgroupBarrier = {
    .name = "__acpp_sscp_sub_group_barrier",
    .exactMatch = true,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitSubgroupBarrier(CI, errorStr);
    }
  };
  externalFunctions.push_back(subgroupBarrier);

  ExternalFunctionInfo workgroupScanInfo = {
    .name = "__acpp_sscp_work_group_inclusive_scan",
    .exactMatch = false,
    .deps = {
      "__acpp_sscp_work_group_inclusive",
    },
    .needsLocalMemory = true,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitWorkgroupScanCall(CI, errorStr);
    },
  };
  externalFunctions.push_back(workgroupScanInfo);

  ExternalFunctionInfo workgroupExclusiveScanInfo = {
    .name = "__acpp_sscp_work_group_exclusive_scan",
    .exactMatch = false,
    .deps = {
      "__acpp_sscp_work_group_exclusive",
    },
    .needsLocalMemory = true,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitWorkgroupExclusiveScanCall(CI, errorStr);
    },
  };
  externalFunctions.push_back(workgroupExclusiveScanInfo);

  ExternalFunctionInfo workgroupReduceInfo = {
    .name = "__acpp_sscp_work_group_reduce",
    .exactMatch = false,
    .deps = {
      "__acpp_sscp_work_group_reduce_helper",
    },
    .needsLocalMemory = true,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitWorkgroupReduceCall(CI, errorStr);
    },
  };
  externalFunctions.push_back(workgroupReduceInfo);

  for (auto& funcInfo : externalFunctions) {
    map[funcInfo.name] = &funcInfo;
  }
}

std::optional<std::string> ExternalFunctionMapper::emitAtomicCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.rfind("__acpp_sscp_atomic_", 0) != 0) {
    errorStr = "Not an atomic function";
    return std::nullopt;
  }

  auto getConstU64 = [&](unsigned idx) -> std::optional<uint64_t> {
    if (idx >= CI->arg_size()) return std::nullopt;
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(CI->getArgOperand(idx))) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto mapMemOrder = [&](uint64_t acppOrder) -> std::string {
    switch(acppOrder) {
      case 0: return "memory_order_relaxed";
      case 1: return "memory_order_consume";
      case 2: return "memory_order_acquire";
      case 3: return "memory_order_release";
      case 4: return "memory_order_acq_rel";
      case 5: return "memory_order_seq_cst";
      default: return "memory_order_relaxed";
    }
  };

  auto clampForModify = [&](const std::string& /*order*/) -> std::string {
    // Metal "modify" atomics are effectively relaxed-only in practice.
    return "memory_order_relaxed";
  };

  // ACPP ABI: first 3 args are (addrSpace, memOrder, memScope) constants
  if (CI->arg_size() < 4) {
    errorStr = "ACPP atomic: too few arguments (need at least 4).";
    return std::nullopt;
  }

  auto addrSpaceC = getConstU64(0); // unused
  auto memOrderC  = getConstU64(1);
  auto memScopeC  = getConstU64(2); // unused

  if (!memOrderC) {
    errorStr = "ACPP atomic: mem_order must be ConstantInt.";
    return std::nullopt;
  }

  llvm::Value* objPtr = CI->getArgOperand(3);
  const std::string addressSpace = addrSpaceMapper(objPtr);
  if (addressSpace != "device" && addressSpace != "threadgroup") {
    errorStr = "ACPP atomic: unsupported address space '" + addressSpace + "' for MSL atomics (expected device/threadgroup).";
    return std::nullopt;
  }

  const std::string order = mapMemOrder(*memOrderC);

  llvm::Type* elemTy = CI->getType();
  if (elemTy->isVoidTy()) {
    if (CI->arg_size() < 5) {
      errorStr = "ACPP atomic store: missing value argument.";
      return std::nullopt;
    }
    elemTy = CI->getArgOperand(4)->getType();
  }

  std::string typeName, typeNameSigned;
  if (elemTy->isIntegerTy(32)) {
    typeName = "uint";
    typeNameSigned = "int";
  } else if (elemTy->isIntegerTy(64)) {
    typeName = "ulong";
    typeNameSigned = "long";
    errorStr = "MSL does not support i64 atomic<T>";
    return std::nullopt;
  } else if (elemTy->isFloatTy()) {
    typeName = "float";
    if (addressSpace != "device") {
      errorStr = "MSL does not support float atomic<T> in threadgroup address space.";
      return std::nullopt;
    }
  } else if (elemTy->isIntegerTy(8) || elemTy->isIntegerTy(16)) {
    errorStr = "MSL does not support i8/i16 atomic<T>. Use i32/i64/float/bool or implement CAS-masked emulation.";
    return std::nullopt;
  } else if (elemTy->isDoubleTy()) {
    errorStr = "MSL does not support double atomic<T>.";
    return std::nullopt;
  } else if (elemTy->isIntegerTy(1)) {
    typeName = "bool";
  } else {
    errorStr = "ACPP atomic: unsupported element type for MSL atomics.";
    return std::nullopt;
  }

  const std::string objCast = "(" + addressSpace + " atomic<" + typeName + "> *)" + exprMapper(objPtr);
  const std::string objCastSigned = "(" + addressSpace + " atomic<" + typeNameSigned + "> *)" + exprMapper(objPtr);

  if (name.find("atomic_load") != std::string::npos) {
    return "atomic_load_explicit(" + objCast + ", " + order + ")";
  }

  if (name.find("atomic_store") != std::string::npos) {
    if (CI->arg_size() < 5) {
      errorStr = "ACPP atomic_store: missing value argument.";
      return std::nullopt;
    }
    std::string val = exprMapper(CI->getArgOperand(4));
    return "atomic_store_explicit(" + objCast + ", " + val + ", " + order + ")";
  }

  if (name.find("atomic_exchange") != std::string::npos) {
    if (CI->arg_size() < 5) {
      errorStr = "ACPP atomic_exchange: missing desired argument.";
      return std::nullopt;
    }
    std::string desired = exprMapper(CI->getArgOperand(4));
    std::string modOrder = clampForModify(order);
    return "atomic_exchange_explicit(" + objCast + ", " + desired + ", " + modOrder + ")";
  }

  if (elemTy->isFloatTy()) {
    if (name.find("atomic_fetch_min") != std::string::npos || name.find("atomic_fetch_max") != std::string::npos) {
      if (CI->arg_size() < 5) {
        errorStr = "ACPP atomic_fetch_min: missing operand argument.";
        return std::nullopt;
      }
      std::string operand = exprMapper(CI->getArgOperand(4));
      std::string modOrder = clampForModify(order);
      if (name.find("atomic_fetch_max") != std::string::npos) {
        return "__acpp_atomic_fetch_max(" + objCast + ", " + operand + ")";
      } else {
        return "__acpp_atomic_fetch_min(" + objCast + ", " + operand + ")";
      }
    }
  }

  auto emitFetch = [&](const char* opName) -> std::optional<std::string> {
    if (name.find(opName) == std::string::npos) return std::nullopt;
    if (CI->arg_size() < 5) {
      errorStr = std::string("ACPP ") + opName + ": missing operand argument.";
      return std::nullopt;
    }
    std::string operand = exprMapper(CI->getArgOperand(4));
    std::string modOrder = clampForModify(order);

    // Map opName -> MSL function name
    // e.g. "__acpp_sscp_atomic_fetch_add" -> "atomic_fetch_add_explicit"
    std::string msl = "atomic_fetch_";
    std::string opname = opName;
    auto pos = opname.find("atomic_fetch_");
    if (pos != std::string::npos) {
      opname = opname.substr(pos + std::string("atomic_fetch_").size());
    }
    msl += opname;
    msl += "_explicit";

    if (name.rfind("i32") == name.size() - 3 || name.rfind("i64") == name.size() - 3) {
      // signed variant
      return msl + "(" + objCastSigned + ", " + operand + ", " + modOrder + ")";
    }

    return msl + "(" + objCast + ", " + operand + ", " + modOrder + ")";
  };

  if (auto r = emitFetch("atomic_fetch_add")) { return r; }
  if (auto r = emitFetch("atomic_fetch_sub")) { return r; }
  if (auto r = emitFetch("atomic_fetch_and")) { return r; }
  if (auto r = emitFetch("atomic_fetch_or"))  { return r; }
  if (auto r = emitFetch("atomic_fetch_xor")) { return r; }
  if (auto r = emitFetch("atomic_fetch_min")) { return r; }
  if (auto r = emitFetch("atomic_fetch_max")) { return r; }

  errorStr = "Unknown ACPP atomic op: " + name;

  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitMemoryFenceCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name != "__acpp_sscp_memory_fence") {
    errorStr = "Not a memory fence function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_memory_fence: expected (scope, order)";
    return std::nullopt;
  }

  auto getConstU64 = [&](unsigned idx) -> std::optional<uint64_t> {
    if (idx >= CI->arg_size()) return std::nullopt;
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(CI->getArgOperand(idx))) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto mapMemOrder = [&](uint64_t acppOrder) -> std::string {
    switch(acppOrder) {
      case 0: return "memory_order_relaxed";
      case 1: return "memory_order_consume";
      case 2: return "memory_order_acquire";
      case 3: return "memory_order_release";
      case 4: return "memory_order_acq_rel";
      case 5: return "memory_order_seq_cst";
      default: return "memory_order_seq_cst";
    }
  };

  auto mapThreadScope = [&](uint64_t acppScope) -> std::string {
    switch(acppScope) {
      case 0: return "thread_scope_thread";
      case 1: return "thread_scope_simdgroup";     // if not available, map to thread
      case 2: return "thread_scope_threadgroup";
      case 3: return "thread_scope_device";
      case 4: return "thread_scope_system";
      default: return "thread_scope_device";
    }
  };

  auto scopeC = getConstU64(0);
  auto orderC = getConstU64(1);
  if (!scopeC) {
    errorStr = "__acpp_sscp_memory_fence: scope must be ConstantInt";
    return std::nullopt;
  }
  if (!orderC) {
    errorStr = "__acpp_sscp_memory_fence: order must be ConstantInt";
    return std::nullopt;
  }

  const std::string order = mapMemOrder(*orderC);
  const std::string scope = mapThreadScope(*scopeC);

  std::string flags = "mem_flags::mem_device";

  return "atomic_thread_fence(" + flags + ", " + order + ", " + scope + ")";
}

std::optional<std::string> ExternalFunctionMapper::emitSubgroupScanCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_sub_group_inclusive_scan") == std::string::npos) {
    errorStr = "Not a subgroup inclusive scan function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_sub_group_inclusive_scan: expected (value, op)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_sub_group_inclusive_scan: op must be ConstantInt";
    return std::nullopt;
  }

  if (*opC == 0) {
    // plus
    return "simd_prefix_inclusive_sum(" + valExpr + ")";
  }
  if (*opC == 1) {
    // multiply
    return "simd_prefix_inclusive_product(" + valExpr + ")";
  }

  errorStr = "Unsupported op for __acpp_sscp_sub_group_inclusive_scan";
  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitSubgroupExclusiveScanCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_sub_group_exclusive_scan") == std::string::npos) {
    errorStr = "Not a subgroup exclusive scan function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_sub_group_exclusive_scan: expected (value, op)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_sub_group_exclusive_scan: op must be ConstantInt";
    return std::nullopt;
  }

  if (*opC == 0) {
    // plus
    return "simd_prefix_exclusive_sum(" + valExpr + ")";
  }
  if (*opC == 1) {
    // multiply
    return "simd_prefix_exclusive_product(" + valExpr + ")";
  }

  errorStr = "Unsupported op for __acpp_sscp_sub_group_exclusive_scan";
  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitSubgroupReduceCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_sub_group_reduce") == std::string::npos) {
    errorStr = "Not a subgroup reduce function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_sub_group_reduce: expected (value, op)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_sub_group_reduce: op must be ConstantInt";
    return std::nullopt;
  }

  if (*opC == 0) {
    // plus
    return "simd_sum(" + valExpr + ")";
  }
  if (*opC == 1) {
    // multiply
    return "simd_product(" + valExpr + ")";
  }

  errorStr = "Unsupported op for __acpp_sscp_sub_group_reduce";
  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitSubgroupBarrier(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name != "__acpp_sscp_sub_group_barrier") {
    errorStr = "Not a subgroup barrier function";
    return std::nullopt;
  }

  llvm::Value* scope = CI->getArgOperand(0);
  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto scopeC = getConstU64(scope);
  if (*scopeC <= 1) {
    return "simdgroup_barrier(mem_flags::mem_none)";
  }
  if (*scopeC == 2) {
    return "simdgroup_barrier(mem_flags::mem_threadgroup)";
  }

  return "simdgroup_barrier(mem_flags::mem_device)";
}

std::optional<std::string> ExternalFunctionMapper::emitWorkgroupScanCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_work_group_inclusive_scan") == std::string::npos) {
    errorStr = "Not a workgroup inclusive scan function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_work_group_inclusive_scan: expected (op, value)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_work_group_inclusive_scan: op must be ConstantInt";
    return std::nullopt;
  }

  auto type = typeMapper(valArg->getType());

  if (*opC == 0) {
    // plus
    return "__acpp_sscp_work_group_inclusive<0>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  if (*opC == 1) {
    // multiply
    return "__acpp_sscp_work_group_inclusive<1>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  errorStr = "Unsupported op for __acpp_sscp_work_group_inclusive_scan";
  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitWorkgroupExclusiveScanCall(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_work_group_exclusive_scan") == std::string::npos) {
    errorStr = "Not a workgroup exclusive scan function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_work_group_exclusive_scan: expected (op, value)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_work_group_exclusive_scan: op must be ConstantInt";
    return std::nullopt;
  }

  auto type = typeMapper(valArg->getType());

  if (*opC == 0) {
    // plus
    return "__acpp_sscp_work_group_exclusive<0>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  if (*opC == 1) {
    // multiply
    return "__acpp_sscp_work_group_exclusive<1>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  errorStr = "Unsupported op for __acpp_sscp_work_group_exclusive_scan";
  return std::nullopt;
}

std::optional<std::string> ExternalFunctionMapper::emitWorkgroupReduceCall(const llvm::CallInst* CI, std::string& errorStr)
{
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }

  const std::string name = F->getName().str();
  if (name.find("__acpp_sscp_work_group_reduce") == std::string::npos) {
    errorStr = "Not a workgroup reduce function";
    return std::nullopt;
  }

  if (CI->arg_size() < 2) {
    errorStr = "__acpp_sscp_work_group_reduce: expected (op, value)";
    return std::nullopt;
  }

  llvm::Value* opArg = CI->getArgOperand(0);
  llvm::Value* valArg = CI->getArgOperand(1);

  std::string valExpr = exprMapper(valArg);

  auto getConstU64 = [&](llvm::Value* V) -> std::optional<uint64_t> {
    if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
      return C->getZExtValue();
    }
    return std::nullopt;
  };

  auto opC = getConstU64(opArg);
  if (!opC) {
    errorStr = "__acpp_sscp_work_group_reduce: op must be ConstantInt";
    return std::nullopt;
  }

  auto type = typeMapper(valArg->getType());

  if (*opC == 0) {
    // plus
    return "__acpp_sscp_work_group_reduce_helper<0>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  if (*opC == 1) {
    // multiply
    return "__acpp_sscp_work_group_reduce_helper<1>((" + type + ")" + valExpr + ", " + "(threadgroup " + type + "*)" + "__acpp_sscp_get_dynamic_local_memory, __acpp_sscp_dynamic_local_memory_size)";
  }

  errorStr = "Unsupported op for __acpp_sscp_work_group_reduce";
  return std::nullopt;
}

} // namespace hipsycl
} // namespace compiler