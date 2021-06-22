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

#ifndef HIPSYCL_VARIABLEUNIFORMITYANALYSIS_HPP
#define HIPSYCL_VARIABLEUNIFORMITYANALYSIS_HPP

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/LegacyPassManagers.h>
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

/**
 * Analyses the variables in the function to figure out if a variable
 * value is
 *
 * a) 'uniform', i.e., always same for all work-items in the *same work-group*
 * b) 'varying', i.e., somehow dependent on the work-item id
 *
 * For safety, 'variable' is assumed, unless certain of a).
 *
 * VAU is an "accumulating" pass; it gathers uniformity information of
 * instructions in a way that it needs not to be invalidated even though
 * the CFG is modified. Thus, in case the semantics of the original
 * information does not change, it is safe for passes to set this pass
 * preserved even though new instructions are added or the CFG manipulated.
 */
class VariableUniformityInfo {
  typedef std::map<llvm::Value *, bool> UniformityIndex;
  typedef std::map<llvm::Function *, UniformityIndex> UniformityCache;
  mutable UniformityCache UniformityCache_;

  bool isUniformityAnalyzed(llvm::Function *F, llvm::Value *V) const;

public:
  explicit VariableUniformityInfo();

  void analyzeFunction(llvm::Function &F, llvm::LoopInfo &LI, llvm::PostDominatorTree &PDT);

  bool isUniform(llvm::Function *F, llvm::Value *V);
  void setUniform(llvm::Function *F, llvm::Value *V, bool IsUniform = true);
  void analyzeBBDivergence(llvm::Function *F, llvm::BasicBlock *BB, llvm::BasicBlock *PreviousUniformBB,
                           llvm::PostDominatorTree &PDT);

  bool shouldBePrivatized(llvm::Function *F, llvm::Value *V);
  bool doFinalization(llvm::Module &M);
  void markInductionVariables(llvm::Function &F, llvm::Loop &L);

  bool invalidate(llvm::Function &F, const llvm::PreservedAnalyses &PA, llvm::FunctionAnalysisManager::Invalidator &) {
    return false;
  }
};

class VariableUniformityAnalysisLegacy : public llvm::FunctionPass {
  llvm::Optional<VariableUniformityInfo> VariableUniformity_;

public:
  static char ID;

  explicit VariableUniformityAnalysisLegacy() : llvm::FunctionPass(ID) {}

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  VariableUniformityInfo &getResult() { return *VariableUniformity_; }
};

class VariableUniformityAnalysis : public llvm::AnalysisInfoMixin<VariableUniformityAnalysis> {
  friend llvm::AnalysisInfoMixin<VariableUniformityAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = VariableUniformityInfo;

  Result run(llvm::Function &F, llvm::FunctionAnalysisManager &AM);
};

} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_VARIABLEUNIFORMITYANALYSIS_HPP
