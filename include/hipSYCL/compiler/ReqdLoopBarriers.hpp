/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
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

#ifndef HIPSYCL_REQDLOOPBARRIERS_HPP
#define HIPSYCL_REQDLOOPBARRIERS_HPP

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace hipsycl {
namespace compiler {

class AddRequiredLoopBarriersPassLegacy : public llvm::LoopPass {
public:
  static char ID;

  explicit AddRequiredLoopBarriersPassLegacy() : llvm::LoopPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL required loop barrier adding pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) override;
};

class AddRequiredLoopBarriersPass : public llvm::PassInfoMixin<AddRequiredLoopBarriersPass> {
public:
  explicit AddRequiredLoopBarriersPass() {}

  llvm::PreservedAnalyses run(llvm::Loop &L, llvm::LoopAnalysisManager &AM, llvm::LoopStandardAnalysisResults &AR,
                              llvm::LPMUpdater &LPMU);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_REQDLOOPBARRIERS_HPP
