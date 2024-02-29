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

#include "hipSYCL/compiler/cbs/LoopsParallelMarker.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/IR/Dominators.h>

namespace {
using namespace hipsycl::compiler;
void markLoopParallel(llvm::Function &F, llvm::Loop *L) {
#if LLVM_VERSION_MAJOR > 12 ||                                                                     \
    (LLVM_VERSION_MAJOR == 12 && LLVM_VERSION_MINOR == 0 && LLVM_VERSION_PATCH == 1)
  // LLVM < 12.0.1 might miscompile if conditionals in "parallel" loop (https://llvm.org/PR46666)

  // Mark memory accesses with access group
  auto *MDAccessGroup = llvm::MDNode::getDistinct(F.getContext(), {});
  for (auto *BB : L->blocks()) {
    for (auto &I : *BB) {
      if (I.mayReadOrWriteMemory() && !I.hasMetadata(llvm::LLVMContext::MD_access_group)) {
        utils::addAccessGroupMD(&I, MDAccessGroup);
      }
    }
  }

  // make the access group parallel w.r.t the WI loop
  utils::createParallelAccessesMdOrAddAccessGroup(&F, L, MDAccessGroup);

  // debug, check whether loop is really marked parallel.
  if (HIPSYCL_DEBUG_LEVEL_INFO <= hipsycl::common::output_stream::get().get_debug_level()) {
    if (utils::isAnnotatedParallel(L)) {
      HIPSYCL_DEBUG_INFO << "[ParallelMarker] loop is parallel: " << L->getHeader()->getName()
                         << "\n";
    } else if (L->getLoopID()) {
      assert(L->getLoopID());
      const llvm::Module *M = F.getParent();
      HIPSYCL_DEBUG_WARNING << "[ParallelMarker] failed to mark wi-loop as parallel, loop id for "
                            << L->getHeader()->getName();
      HIPSYCL_DEBUG_EXECUTE_WARNING(
        L->getLoopID()->print(llvm::outs(), M);
        for (auto &MDOp : llvm::drop_begin(L->getLoopID()->operands(), 1)) {
          MDOp->print(llvm::outs(), M);
        }
        llvm::outs() << "\n";
      )
    }
  }
#endif
}

void addVectorizationHints(const llvm::Function &F, const llvm::TargetTransformInfo &TTI,
                           const llvm::Loop *L) {
  llvm::SmallVector<llvm::MDNode *, 3> PostTransformMD;
  // work-item loops should be vectorizable, so emit metadata to suggest so
  if (!llvm::findOptionMDForLoop(L, "llvm.loop.vectorize.enable")) {
    auto *MDVectorize = llvm::MDNode::get(
        F.getContext(), {llvm::MDString::get(F.getContext(), "llvm.loop.vectorize.enable"),
                         llvm::ConstantAsMetadata::get(llvm::Constant::getAllOnesValue(
                             llvm::IntegerType::get(F.getContext(), 1)))});
    PostTransformMD.push_back(MDVectorize);
  }
#if LLVM_VERSION_MAJOR >= 12
  // enable scalable vectorization
  if (TTI.supportsScalableVectors()) {
    if (!llvm::findOptionMDForLoop(L, "llvm.loop.vectorize.scalable.enable")) {
      auto *MDVectorize = llvm::MDNode::get(
          F.getContext(),
          {llvm::MDString::get(F.getContext(), "llvm.loop.vectorize.scalable.enable"),
           llvm::ConstantAsMetadata::get(
               llvm::Constant::getAllOnesValue(llvm::IntegerType::get(F.getContext(), 1)))});
      PostTransformMD.push_back(MDVectorize);
    }
  }
#endif

  if (!PostTransformMD.empty()) {
    auto *LoopID =
        llvm::makePostTransformationMetadata(F.getContext(), L->getLoopID(), {}, PostTransformMD);
    L->setLoopID(LoopID);
  }
}

bool markLoopsWorkItem(llvm::Function &F, const llvm::LoopInfo &LI,
                       const llvm::TargetTransformInfo &TTI) {
  bool Changed = false;

  for (auto *SL : utils::getLoopsInPreorder(LI)) {
    if (utils::isWorkItemLoop(*SL)) {
      Changed = true;
      HIPSYCL_DEBUG_INFO << "[ParallelMarker] Mark loop: " << SL->getName() << "\n";

      markLoopParallel(F, SL);

      addVectorizationHints(F, TTI, SL);
    }
  }

  if (F.hasFnAttribute(llvm::Attribute::NoInline) &&
      !F.hasFnAttribute(llvm::Attribute::OptimizeNone))
    F.removeFnAttr(llvm::Attribute::NoInline);

  if (!Changed) {
    HIPSYCL_DEBUG_INFO << "[ParallelMarker] no wi loop found..?\n";
  }
  return Changed;
}
} // namespace

void hipsycl::compiler::LoopsParallelMarkerPassLegacy::getAnalysisUsage(
    llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::TargetTransformInfoWrapperPass>();
  AU.addPreserved<llvm::TargetTransformInfoWrapperPass>();
}
bool hipsycl::compiler::LoopsParallelMarkerPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  const auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  const auto &TTI = getAnalysis<llvm::TargetTransformInfoWrapperPass>().getTTI(F);

  return markLoopsWorkItem(F, LI, TTI);
}

llvm::PreservedAnalyses
hipsycl::compiler::LoopsParallelMarkerPass::run(llvm::Function &F,
                                                llvm::FunctionAnalysisManager &AM) {
  const auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  const auto &TTI = AM.getResult<llvm::TargetIRAnalysis>(F);
  if (!SAA) {
    llvm::errs() << "SplitterAnnotationAnalysis not cached.\n";
    return llvm::PreservedAnalyses::all();
  }
  if (SAA->isKernelFunc(&F))
    markLoopsWorkItem(F, LI, TTI);

  return llvm::PreservedAnalyses::all();
}

char hipsycl::compiler::LoopsParallelMarkerPassLegacy::ID = 0;
