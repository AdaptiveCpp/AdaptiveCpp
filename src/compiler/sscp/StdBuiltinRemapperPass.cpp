/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2023 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/sscp/StdBuiltinRemapperPass.hpp"
#include "hipSYCL/common/debug.hpp"
#include <unordered_set>
#include <unordered_map>

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

llvm::PreservedAnalyses StdBuiltinRemapperPass::run(llvm::Module &M,
                                                    llvm::ModuleAnalysisManager &MAM) {

  std::unordered_set<std::string> BuiltinsToReplace;
  for(const char* name : math_builtins) {
    BuiltinsToReplace.insert(std::string{name});
  }

  auto ParseBuiltinName = [&](const std::string &Name, std::string &BaseNameOut,
                              std::string &SuffixOut) -> bool {
    if(BuiltinsToReplace.count(Name+"f") > 0) {
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

  for(const auto& B : BuiltinsToReplace) {
    std::string Suffix;
    std::string BaseName;
    if(ParseBuiltinName(B, BaseName, Suffix)) {
      Replacements[B] = "__hipsycl_sscp_" + BaseName + "_" + Suffix;
    }
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

  return llvm::PreservedAnalyses::none();
}
}
}

