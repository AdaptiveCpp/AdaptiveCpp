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

#ifndef HIPSYCL_SPLITTERANNOTATIONANALYSIS_HPP
#define HIPSYCL_SPLITTERANNOTATIONANALYSIS_HPP

#include <llvm/IR/LegacyPassManagers.h>
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

// collects all functions annotated as nd-range kernels or barriers.
class SplitterAnnotationInfo {
  static constexpr const char *SplitterAnnotation = "hipsycl_barrier";
  static constexpr const char *KernelAnnotation = "hipsycl_nd_kernel";
  llvm::SmallPtrSet<llvm::Function *, 4> SplitterFuncs;
  llvm::SmallPtrSet<llvm::Function *, 8> NDKernels;

  bool analyzeModule(llvm::Module &M);

public:
  explicit SplitterAnnotationInfo(llvm::Module &Module);
  inline bool isSplitterFunc(const llvm::Function *F) const { return SplitterFuncs.contains(F); }
  inline bool isKernelFunc(const llvm::Function *F) const { return NDKernels.contains(F); }

  inline void removeSplitter(llvm::Function *F) { SplitterFuncs.erase(F); }
  inline void addSplitter(llvm::Function *F) { SplitterFuncs.insert(F); }

  void print(llvm::raw_ostream &Stream);

  bool invalidate(llvm::Module &, const llvm::PreservedAnalyses &,
                  llvm::ModuleAnalysisManager::Invalidator &) {
    return false;
  }
};

/*!
 * \brief Searches the module for functions annotated with \b hipsycl_splitter.
 *
 * \note Needs to be a FunctionPass, as we depend on this from the LoopPass.
 *       If it was a ModulePass we would end up in a non-preserving loop of death..
 *       As the annotations should not change from call to call, we cache the result in an Optional.
 */
class SplitterAnnotationAnalysisLegacy : public llvm::FunctionPass {
  llvm::Optional<SplitterAnnotationInfo> SplitterAnnotation_;

public:
  static char ID;

  explicit SplitterAnnotationAnalysisLegacy() : llvm::FunctionPass(ID) {}

  // todo: evaluate whether we want to reevaluate...
  // void releaseMemory() override { splitterAnnotation_.reset(); }

  bool runOnFunction(llvm::Function &F) override;
  void getAnalysisUsage(llvm::AnalysisUsage &AU) const override { AU.setPreservesAll(); }

  const SplitterAnnotationInfo &getAnnotationInfo() const { return *SplitterAnnotation_; }
  SplitterAnnotationInfo &getAnnotationInfo() { return *SplitterAnnotation_; }
};

/*!
 * \brief Searches the module for functions annotated with \b hipsycl_splitter (new PM).
 */
class SplitterAnnotationAnalysis : public llvm::AnalysisInfoMixin<SplitterAnnotationAnalysis> {
  friend llvm::AnalysisInfoMixin<SplitterAnnotationAnalysis>;
  static llvm::AnalysisKey Key;

public:
  using Result = SplitterAnnotationInfo;

  Result run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
};

/*!
 * \brief Requires the SplitterAnnotationAnalysis once, so it is actually performed and thus cached
 * by the MAM.
 */
class SplitterAnnotationAnalysisCacher
    : public llvm::PassInfoMixin<SplitterAnnotationAnalysisCacher> {
public:
  explicit SplitterAnnotationAnalysisCacher() {}

  llvm::PreservedAnalyses run(llvm::Module &M, llvm::ModuleAnalysisManager &AM);
  static bool isRequired() { return true; }
};
} // namespace compiler
} // namespace hipsycl
#endif // HIPSYCL_SPLITTERANNOTATIONANALYSIS_HPP
