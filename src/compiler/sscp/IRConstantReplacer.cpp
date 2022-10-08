/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include <climits>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <type_traits>

namespace hipsycl {
namespace compiler {

S1IRConstantReplacer::S1IRConstantReplacer(
    const std::unordered_map<std::string, int> &IntConstants,
    const std::unordered_map<std::string, uint64_t> &UInt64Constants,
    const std::unordered_map<std::string, std::string> &StringConstants)
    : IntConstants{IntConstants}, UInt64Constants{UInt64Constants}, StringConstants{
                                                                        StringConstants} {}

llvm::PreservedAnalyses S1IRConstantReplacer::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {
  auto setConstants = [&](const auto& ConstantReplacementTable) {
    for(const auto& IC : ConstantReplacementTable) {
      if(llvm::GlobalVariable* G = M.getGlobalVariable(IC.first, true)) {
        IRConstant C{M, *G};
        C.set(IC.second);
      }
    }
  };

  setConstants(IntConstants);
  setConstants(UInt64Constants);
  setConstants(StringConstants);


  // TODO Make this more specific
  return llvm::PreservedAnalyses::none();
}
}
}
