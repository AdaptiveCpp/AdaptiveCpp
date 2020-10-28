/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019 Aksel Alpay and contributors
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

#ifndef HIPSYCL_LOOPSPLITTER_HPP
#define HIPSYCL_LOOPSPLITTER_HPP

#include "llvm/Analysis/LoopPass.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"

namespace hipsycl {
namespace compiler {

/*!
 * \brief Searches the module for functions annotated with \b hipsycl_splitter.
 *
 * \note Needs to be a FunctionPass, as we depend on this from the LoopPass.
 *       If it was a ModulePass we would end up in a non-preserving loop of death..
 *       As the annotations should not change from call to call, we cache the result in an Optional.
 */
class SplitterAnnotationAnalysisLegacy : public llvm::FunctionPass {
  llvm::Optional<llvm::SmallPtrSet<llvm::Function *, 2>> splitterFuncs_;
  static constexpr char *SplitterAnnotation = "hipsycl_splitter";

public:
  static char ID;

  explicit SplitterAnnotationAnalysisLegacy() : llvm::FunctionPass(ID) {}

  // todo: evaluate whether we want to reevaluate...
  // void releaseMemory() override { splitterFuncs_.reset(); }

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override { AU.setPreservesAll(); }

  bool IsSplitterFunc(llvm::Function *F) const { return splitterFuncs_->find(F) != splitterFuncs_->end(); }
};

class LoopSplitAtBarrierPassLegacy : public llvm::LoopPass {

public:
  static char ID;

  explicit LoopSplitAtBarrierPassLegacy() : llvm::LoopPass(ID) {}

  llvm::StringRef getPassName() const override { return "hipSYCL loop splitting pass"; }

  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override;

  bool runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) override;
};
} // namespace compiler
} // namespace hipsycl

#endif