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
#ifndef HIPSYCL_SPLITTERANNOTATIONANALYSIS_HPP
#define HIPSYCL_SPLITTERANNOTATIONANALYSIS_HPP

#include <llvm/IR/LegacyPassManagers.h>
#include <llvm/IR/PassManager.h>

#include <optional>

namespace hipsycl {
namespace compiler {

// collects all functions annotated as nd-range kernels or barriers.
class SplitterAnnotationInfo {
  static constexpr const char *SplitterAnnotation = "hipsycl_barrier";
  static constexpr const char *KernelAnnotation = "hipsycl_nd_kernel";
  static constexpr const char *SSCPKernelMD = "kernel";
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
  std::optional<SplitterAnnotationInfo> SplitterAnnotation_;

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
