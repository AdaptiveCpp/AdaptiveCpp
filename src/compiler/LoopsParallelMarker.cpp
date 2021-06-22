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

#include "hipSYCL/compiler/LoopsParallelMarker.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Dominators.h>

namespace {
using namespace hipsycl::compiler;
bool markLoopsWorkItem(llvm::Function &F, const llvm::LoopInfo &LI) {
  bool Changed = false;

  for (auto *L : LI.getTopLevelLoops()) {
    for (auto *SL : L->getLoopsInPreorder()) {
      if (SL->getLoopLatch()->getTerminator()->hasMetadata(MDKind::WorkItemLoop)) {
        Changed = true;

        // Mark memory accesses with access group
        auto *MDAccessGroup = llvm::MDNode::getDistinct(F.getContext(), {});
        for (auto *BB : SL->blocks()) {
          for (auto &I : *BB) {
            if (I.mayReadOrWriteMemory() && !I.hasMetadata(llvm::LLVMContext::MD_access_group)) {
              utils::addAccessGroupMD(&I, MDAccessGroup);
            }
          }
        }

        // make the access group parallel w.r.t the WI loop
        utils::createParallelAccessesMdOrAddAccessGroup(&F, SL, MDAccessGroup);

        // work-item loops should be vectorizable, so emit metadata to suggest so
        if (!llvm::findOptionMDForLoop(SL, "llvm.loop.vectorize.enable")) {
          auto *MDVectorize =
              llvm::MDNode::get(F.getContext(), {llvm::MDString::get(F.getContext(), "llvm.loop.vectorize.enable"),
                                                 llvm::ConstantAsMetadata::get(llvm::Constant::getAllOnesValue(
                                                     llvm::IntegerType::get(F.getContext(), 1)))});
          auto *LoopID = llvm::makePostTransformationMetadata(F.getContext(), SL->getLoopID(), {}, {MDVectorize});
          SL->setLoopID(LoopID);
        }

        if (HIPSYCL_DEBUG_LEVEL_INFO <= hipsycl::common::output_stream::get().get_debug_level()) {
          if (utils::isAnnotatedParallel(SL)) {
            HIPSYCL_DEBUG_INFO << "loop is parallel: " << SL->getHeader()->getName() << "\n";
          } else if (SL->getLoopID()) {
            assert(SL->getLoopID());
            const llvm::Module *M = F.getParent();
            HIPSYCL_DEBUG_WARNING << "loop id for " << SL->getHeader()->getName();
            SL->getLoopID()->print(llvm::outs(), M);
            for (auto &MDOp : llvm::drop_begin(SL->getLoopID()->operands(), 1)) {
              MDOp->print(llvm::outs(), M);
            }
            llvm::outs() << "\n";
          }
        }
      }
    }
  }

  return Changed;
}
} // namespace

void hipsycl::compiler::LoopsParallelMarkerPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
}
bool hipsycl::compiler::LoopsParallelMarkerPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  const auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  return markLoopsWorkItem(F, LI);
}

llvm::PreservedAnalyses hipsycl::compiler::LoopsParallelMarkerPass::run(llvm::Function &F,
                                                                        llvm::FunctionAnalysisManager &AM) {
  const auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  const auto &MAMProxy = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAMProxy.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA) {
    llvm::errs() << "SplitterAnnotationAnalysis not cached.\n";
    return llvm::PreservedAnalyses::all();
  }
  if (SAA->isKernelFunc(&F))
    markLoopsWorkItem(F, LI);

  return llvm::PreservedAnalyses::all();
}

char hipsycl::compiler::LoopsParallelMarkerPassLegacy::ID = 0;
