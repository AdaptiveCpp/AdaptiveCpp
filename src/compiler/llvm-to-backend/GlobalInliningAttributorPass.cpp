/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/llvm-to-backend/GlobalInliningAttributorPass.hpp"

namespace hipsycl {
namespace compiler {

GlobalInliningAttributorPass::GlobalInliningAttributorPass(const std::vector<std::string> &KN)
    : KernelNames{KN} {}

llvm::PreservedAnalyses GlobalInliningAttributorPass::run(llvm::Module &M,
                                                          llvm::ModuleAnalysisManager &MAM) {

  llvm::SmallPtrSet<llvm::Function*, 16> Kernels;

  for(auto& KN : KernelNames)
    if(auto* F = M.getFunction(KN))
      Kernels.insert(F);

  for(auto &F : M) {
    // Ignore kernels and intrinsics
    if (!F.isIntrinsic() && !Kernels.contains(&F)) {
      // Ignore undefined functions
      if(!F.empty()) {
        F.setLinkage(llvm::GlobalValue::InternalLinkage);
        if(!F.hasFnAttribute(llvm::Attribute::AlwaysInline))
          F.addFnAttr(llvm::Attribute::AlwaysInline);
      }
    }
  }
  return llvm::PreservedAnalyses::none();
}
}
}


