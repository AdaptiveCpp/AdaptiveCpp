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
#include "hipSYCL/compiler/sscp/StdBuiltinRemapperPass.hpp"
#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include <unordered_set>
#include <unordered_map>

#define STRINGIFY2(x) #x
#define STRINGIFY(x) STRINGIFY2(x)


namespace hipsycl {
namespace compiler {

// List of builtins taken from https://libc.llvm.org/math/
static constexpr std::array math_builtins = {
    "ceil",       "ceilf",       "ceill",       "copysign",  "copysignf",  "copysignl",
    "fabs",       "fabsf",       "fabsl",       "fdim",      "fdimf",      "fdiml",
    "floor",      "floorf",      "floorl",      "fmax",      "fmaxf",      "fmaxl",
    "fmin",       "fminf",       "fminl",       "fmod",      "fmodf",      "fmodl",
    "frexp",      "frexpf",      "frexpl",      "ilogb",     "ilogbf",     "ilogbl",
    "ldexp",      "ldexpf",      "ldexpl",      "llrint",    "llrintf",    "llrintl",
    "llround",    "llroundf",    "llroundl",    "logb",      "logbf",      "logbl",
    "lrint",      "lrintf",      "lrintl",      "lround",    "lroundf",    "lroundl",
    "modf",       "modff",       "modfl",       "nan",       "nanf",       "nanl",
    "nearbyint",  "nearbyintf",  "nearbyintl",  "nextafter", "nextafterf", "nextafterl",
    "nexttoward", "nexttowardf", "nexttowardl", "remainder", "remainderf", "remainderl",
    "remquo",     "remquof",     "remquol",     "rint",      "rintf",      "rintl",
    "round",      "roundf",      "roundl",      "scalbn",    "scalbnf",    "scalbnl",
    "trunc",      "truncf",      "truncl",      "acos",      "acosf",      "acosl",
    "acosh",      "acoshf",      "acoshl",      "asin",      "asinf",      "asinl",
    "asinh",      "asinhf",      "asinhl",      "atan",      "atanf",      "atanl",
    "atan2",      "atan2f",      "atan2l",      "atanh",     "atanhf",     "atanhl",
    "cbrt",       "cbrtf",       "cbrtl",       "cos",       "cosf",       "cosl",
    "cosh",       "coshf",       "coshl",       "erf",       "erff",       "erfl",
    "erfc",       "erfcf",       "erfcl",       "exp",       "expf",       "expl",
    "exp10",      "exp10f",      "exp10l",      "exp2",      "exp2f",      "exp2l",
    "expm1",      "expm1f",      "expm1l",      "fma",       "fmaf",       "fmal",
    "hypot",      "hypotf",      "hypotl",      "lgamma",    "lgammaf",    "lgammal",
    "log",        "logf",        "logl",        "log10",     "log10f",     "log10l",
    "log1p",      "log1pf",      "log1pl",      "log2",      "log2f",      "log2l",
    "pow",        "powf",        "powl",        "sin",       "sinf",       "sinl",
    "sincos",     "sincosf",     "sincosl",     "sinh",      "sinhf",      "sinhl",
    "sqrt",       "sqrtf",       "sqrtl",       "tan",       "tanf",       "tanl",
    "tanh",       "tanhf",       "tanhl",       "tgamma",    "tgammaf",    "tgammal"};

// these are remapped for f32 and f64
static constexpr std::array remapped_llvm_math_builtins = {
  "sin", "cos", "sqrt"
};

using builtin_mapping = std::array<const char*, 2>;
// We may want to complete this with soft-float functions defined here:
// https://gcc.gnu.org/onlinedocs/gccint/Soft-float-library-routines.html
static constexpr std::array explicitly_mapped_builtins = {
  // clang sometimes (e.g. -ffast-math) these builtins
  builtin_mapping{"__powisf2", "__acpp_sscp_pown_f32"},
  builtin_mapping{"__powidf2", "__acpp_sscp_pown_f64"},

#define ACPP_DECLARE_FINITE_BUILTIN_MAPPING(name) \
  builtin_mapping{STRINGIFY(__ ## name ## f_finite), STRINGIFY(__acpp_sscp_ ## name ## _f32)}, \
  builtin_mapping{STRINGIFY(__ ## name ## _finite), STRINGIFY(__acpp_sscp_ ## name ## _f64)}

  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(acos),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(acosh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(asin),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(asinh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(atan2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(atanh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(cosh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(cos),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp10),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(exp),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(fmod),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log10),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log2),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(log),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(hypot),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(pow),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(remainder),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sinh),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sin),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(sqrt),
  ACPP_DECLARE_FINITE_BUILTIN_MAPPING(tan)
};

namespace {

void remapComplexAbs(llvm::Module &M, llvm::StringRef Name,
                     llvm::Type *ScalarTy, llvm::StringRef HypotName) {
  auto *F = M.getFunction(Name);
  if(!F || !F->isDeclaration() || F->use_empty())
    return;

  const unsigned ScalarBits = ScalarTy->getPrimitiveSizeInBits();
  auto IsComponent = [&](llvm::Type *T) {
    return T == ScalarTy || T->isIntegerTy(ScalarBits);
  };
  auto IsPair = [&](llvm::Type *T) {
    if(auto *AT = llvm::dyn_cast<llvm::ArrayType>(T))
      return AT->getNumElements() == 2 && IsComponent(AT->getElementType());
    if(auto *ST = llvm::dyn_cast<llvm::StructType>(T))
      return !ST->isOpaque() && ST->getNumElements() == 2 &&
             IsComponent(ST->getElementType(0)) &&
             IsComponent(ST->getElementType(1));
    return false;
  };

  if(F->isVarArg() || F->getReturnType() != ScalarTy ||
     (F->arg_size() != 1 && F->arg_size() != 2))
    return;

  auto *ArgTy = F->getArg(0)->getType();
  auto *VectorTy = llvm::FixedVectorType::get(ScalarTy, 2);
  const bool Split = F->arg_size() == 2 && IsComponent(ArgTy) &&
                     IsComponent(F->getArg(1)->getType());
  const bool Aggregate = IsPair(ArgTy);
  const bool Packed = ArgTy == VectorTy || ArgTy->isIntegerTy(2 * ScalarBits);
  const bool Indirect = ArgTy->isPointerTy() &&
      !F->hasParamAttribute(0, llvm::Attribute::SwiftError) &&
      !F->hasParamAttribute(0, llvm::Attribute::InAlloca) &&
      !F->hasParamAttribute(0, llvm::Attribute::ByRef) &&
      !F->hasParamAttribute(0, llvm::Attribute::Preallocated) &&
      (!F->hasParamAttribute(0, llvm::Attribute::ByVal) ||
       IsPair(F->getParamByValType(0)));
  if(!Split && !(F->arg_size() == 1 && (Aggregate || Packed || Indirect))) {
    HIPSYCL_DEBUG_WARNING << "StdBuiltinRemapper: Unsupported complex argument ABI for "
                         << Name.str() << "\n";
    return;
  }

  auto *HypotTy = llvm::FunctionType::get(ScalarTy, {ScalarTy, ScalarTy}, false);
  if(auto *Existing = M.getNamedValue(HypotName)) {
    auto *Hypot = llvm::dyn_cast<llvm::Function>(Existing);
    if(!Hypot || Hypot->getFunctionType() != HypotTy ||
       Hypot->getCallingConv() != llvm::CallingConv::C)
      return;
  }

  // Preserve the host ABI at existing call sites, but do not retain the libm
  // name: LLVM could otherwise turn the wrapper back into a library call.
  auto *Wrapper = llvm::Function::Create(F->getFunctionType(),
      llvm::GlobalValue::InternalLinkage, "__acpp_" + Name, M);
  Wrapper->setCallingConv(F->getCallingConv());
  Wrapper->setAttributes(F->getAttributes());
  Wrapper->removeFnAttr(llvm::Attribute::NoInline);
  Wrapper->removeFnAttr(llvm::Attribute::OptimizeNone);
  Wrapper->addFnAttr(llvm::Attribute::AlwaysInline);
  llvm::IRBuilder<> Builder{llvm::BasicBlock::Create(M.getContext(), "entry", Wrapper)};
  llvm::Value *Real;
  llvm::Value *Imag;
  auto *Arg = Wrapper->getArg(0);
  if(Split) {
    Real = Arg;
    Imag = Wrapper->getArg(1);
  } else if(Aggregate) {
    // AArch64 uses arrays; Arm hard/soft-float ABIs use structs/integer arrays.
    Real = Builder.CreateExtractValue(Arg, {0});
    Imag = Builder.CreateExtractValue(Arg, {1});
  } else if(Packed) {
    // x86-64 SysV passes complex float as <2 x float>, Windows as i64.
    auto *Pair = Builder.CreateBitCast(Arg, VectorTy);
    Real = Builder.CreateExtractElement(Pair, Builder.getInt32(0));
    Imag = Builder.CreateExtractElement(Pair, Builder.getInt32(1));
  } else {
    // Indirect arguments (e.g. x86 byval and Windows complex double) store two
    // consecutive components. Do not assume a stronger alignment than the ABI.
    auto *PtrTy = llvm::PointerType::get(ScalarTy, ArgTy->getPointerAddressSpace());
    auto *Ptr = Builder.CreatePointerCast(Arg, PtrTy);
    Real = Builder.CreateAlignedLoad(ScalarTy, Ptr, llvm::Align(1));
    Imag = Builder.CreateAlignedLoad(ScalarTy,
        Builder.CreateConstGEP1_32(ScalarTy, Ptr, 1), llvm::Align(1));
  }
  Real = Builder.CreateBitCast(Real, ScalarTy);
  Imag = Builder.CreateBitCast(Imag, ScalarTy);
  // hypot handles scaling, signed zeros and Inf/NaN without naive squaring.
  auto Hypot = M.getOrInsertFunction(HypotName, HypotTy);
  Builder.CreateRet(Builder.CreateCall(Hypot, {Real, Imag}));
  F->replaceAllUsesWith(Wrapper);
}

}

llvm::PreservedAnalyses StdBuiltinRemapperPass::run(llvm::Module &M,
                                                    llvm::ModuleAnalysisManager &MAM) {

  std::unordered_set<std::string> GenericBuiltinsToReplace;
  for(const char* name : math_builtins) {
    GenericBuiltinsToReplace.insert(std::string{name});
  }

  auto ParseBuiltinName = [&](const std::string &Name, std::string &BaseNameOut,
                              std::string &SuffixOut) -> bool {
    if(GenericBuiltinsToReplace.count(Name+"f") > 0) {
      SuffixOut = "f64";
      BaseNameOut = Name;
    } else {
      if(Name.find("f") == Name.size() - 1) {
        SuffixOut = "f32";
        BaseNameOut = Name.substr(0, Name.size() - 1);
      } else {
        return false;
      }
    }
    return true;
  };

  std::unordered_map<std::string, std::string> Replacements;

  for(const auto& B : GenericBuiltinsToReplace) {
    std::string Suffix;
    std::string BaseName;
    if(ParseBuiltinName(B, BaseName, Suffix)) {
      Replacements[B] = "__acpp_sscp_" + BaseName + "_" + Suffix;
    }
  }
  for(const auto& EM : explicitly_mapped_builtins) {
    Replacements[EM[0]] = EM[1];
  }
  for(const auto& B : remapped_llvm_math_builtins) {
    std::string builtin_name = std::string{B};
    Replacements["llvm." + builtin_name + ".f32"] = "__acpp_sscp_" + builtin_name + "_f32";
    Replacements["llvm." + builtin_name + ".f64"] = "__acpp_sscp_" + builtin_name + "_f64";
  }

  for(const auto& B: Replacements) {
    // Find function to replace
    if(llvm::Function* F = M.getFunction(B.first)) {
      // See if we have replacement declaration
      llvm::Function* Replacement = M.getFunction(B.second);
      // If not, create declaration
      if(!Replacement) {
        Replacement = llvm::Function::Create(F->getFunctionType(), F->getLinkage(), B.second, M);
        Replacement->setLinkage(llvm::GlobalValue::ExternalLinkage);
      }
      if(F->getFunctionType() == Replacement->getFunctionType()) {
        HIPSYCL_DEBUG_INFO << "StdBuiltinRemapper: Remapping calls from " << B.first << " to "
                           << B.second << "\n";
        F->replaceAllUsesWith(Replacement);
      }
    }
  }

  remapComplexAbs(M, "cabsf", llvm::Type::getFloatTy(M.getContext()),
                  "__acpp_sscp_hypot_f32");
  remapComplexAbs(M, "cabs", llvm::Type::getDoubleTy(M.getContext()),
                  "__acpp_sscp_hypot_f64");

  return llvm::PreservedAnalyses::none();
}
}
}

