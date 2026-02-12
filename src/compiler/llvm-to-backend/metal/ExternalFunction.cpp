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
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>

#include <algorithm>
#include <functional>
#include <unordered_set>

namespace hipsycl {
namespace compiler {

namespace {

std::vector<ExternalFunctionInfo> generateLLVMIntrinsics() {
  const std::vector<std::tuple<std::string, std::string, int>> intrinsics = {
    {"ctlz", "clz", 1},
    {"cttz", "ctz", 1},
    {"ctpop", "popcount", -1},
    {"umin", "min", 2},
    {"umax", "max", 2},
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

std::optional<uint64_t> getConstU64(llvm::Value* V) {
  if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(V)) {
    return C->getZExtValue();
  }
  return std::nullopt;
}

std::optional<std::string> extractStringConstant(llvm::Value* V, std::string& errorStr) {
  llvm::GlobalVariable* GV = nullptr;

  // Handle either direct GlobalVariable or ConstantExpr that refers to one
  if (auto* gv = llvm::dyn_cast<llvm::GlobalVariable>(V)) {
    GV = gv;
  } else if (auto* CE = llvm::dyn_cast<llvm::ConstantExpr>(V)) {
    if (CE->getOpcode() == llvm::Instruction::GetElementPtr) {
      GV = llvm::dyn_cast<llvm::GlobalVariable>(CE->getOperand(0));
    }
  }

  if (!GV || !GV->hasInitializer()) {
    errorStr = "Argument must be a string constant";
    return std::nullopt;
  }

  auto* CDA = llvm::dyn_cast<llvm::ConstantDataArray>(GV->getInitializer());
  if (!CDA || !CDA->isString()) {
    errorStr = "Argument must be a string constant";
    return std::nullopt;
  }

  std::string result = CDA->getAsString().str();
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

struct EmitContext {
  const llvm::Function* F;
  std::string name;
};

std::optional<EmitContext> initEmitContext(const llvm::CallInst* CI, std::string& errorStr) {
  errorStr.clear();
  const llvm::Function* F = CI ? CI->getCalledFunction() : nullptr;
  if (!F) {
    errorStr = "CallInst has no called function";
    return std::nullopt;
  }
  return EmitContext{F, F->getName().str()};
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
      "llvm.usub.sat.",
      "__acpp_sscp_usub_sat",
R"__(
  template<typename T>
  inline T __acpp_sscp_usub_sat(T a, T b) {
      return (a < b) ? 0 : (a - b);
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
  };

  auto append = [&](auto&& funcs) {
    externalFunctions.insert(
      externalFunctions.end(),
      std::make_move_iterator(funcs.begin()),
      std::make_move_iterator(funcs.end())
    );
  };

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
  ExternalFunctionInfo metalInlineInfo = {
  .name = "__acpp_sscp_metal",
    .exactMatch = false,
    .customCallEmitter = [this](const llvm::CallInst* CI, std::string& errorStr) -> std::optional<std::string> {
      return emitMetalInlineCall(CI, errorStr);
    }
  };
  externalFunctions.push_back(metalInlineInfo);

  for (auto& funcInfo : externalFunctions) {
    map[funcInfo.name] = &funcInfo;
  }
}

std::optional<std::string> ExternalFunctionMapper::emitMetalInlineCall(const llvm::CallInst* CI, std::string& errorStr) {
  auto ctx = initEmitContext(CI, errorStr);
  if (!ctx) return std::nullopt;

  if (ctx->name.find("__acpp_sscp_metal") == std::string::npos) {
    errorStr = "Not a metal function";
    return std::nullopt;
  }

  bool is_symbol = ctx->name.find("__acpp_sscp_metal_symbol") == 0;

  if (CI->arg_size() < 1) {
    errorStr = "__acpp_sscp_metal: expected at least 1 argument (function name / constant)";
    return std::nullopt;
  }

  if (is_symbol && CI->arg_size() != 1) {
    errorStr = "__acpp_sscp_metal_symbol: expected at least 1 arguments (symbol name constant)";
    return std::nullopt;
  }

  // Extract first argument as string constant (function name)
  auto funcName = extractStringConstant(CI->getArgOperand(0), errorStr);
  if (!funcName) {
    errorStr = "__acpp_sscp_metal: " + errorStr;
    return std::nullopt;
  }

  if (is_symbol) {
    return *funcName;
  }

  std::string result;
  if (funcName->find("%s") != std::string::npos) {
    // expand printf-style format string with arguments
    size_t pos = 0;
    int arg = 1;
    while (pos != std::string::npos && arg < CI->arg_size()) {
      auto next = funcName->find("%s", pos);
      result += funcName->substr(pos, next == std::string::npos ? std::string::npos : next - pos);
      llvm::Value* argValue = CI->getArgOperand(arg++);
      result += exprMapper(argValue);
      pos = next == std::string::npos ? std::string::npos : next + 2;
    }
    result += funcName->substr(pos);
    return result;
  } else {
    result = *funcName + "(";

    for (unsigned i = 1; i < CI->arg_size(); ++i) {
      if (i > 1) {
        result += ", ";
      }
      llvm::Value* arg = CI->getArgOperand(i);
      result += exprMapper(arg);
    }

    result += ")";
  }
  return result;
}

} // namespace hipsycl
} // namespace compiler