// LLVM function pass to convert all PHIs to allocas.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
//               2021 Aksel Alpay and hipSYCL contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "hipSYCL/compiler/cbs/PHIsToAllocas.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>

#include <iostream>

namespace {
/**
 * Convert a PHI to a read from a stack value and all the sources to
 * writes to the same stack value.
 *
 * Used to fix context save/restore issues with regions with PHI nodes in the
 * entry node (usually due to the use of work group scope variables such as
 * B-loop iteration variables). In case of PHI nodes at region entries, we cannot
 * just insert the context restore code because it is assumed there are no
 * non-phi Instructions before PHIs which the context restore
 * code constitutes to. Secondly, in case the PHINode is at a
 * region entry (e.g. a B-Loop) adding new basic blocks before it would
 * break the assumption of single entry regions.
 */
llvm::Instruction *breakPHIToAllocas(llvm::PHINode *Phi) {
  std::string AllocaName{std::string(Phi->getName().str()) + ".ex_phi"};

  llvm::Function *Function = Phi->getParent()->getParent();

  llvm::IRBuilder Builder(&*(Function->getEntryBlock().getFirstInsertionPt()));

  llvm::AllocaInst *Alloca = Builder.CreateAlloca(Phi->getType(), 0, AllocaName);

  for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues(); ++Incoming) {
    auto *V = Phi->getIncomingValue(Incoming);
    auto *IncomingBB = Phi->getIncomingBlock(Incoming);
    Builder.SetInsertPoint(IncomingBB->getTerminator());
    Builder.CreateStore(V, Alloca);
  }
  Builder.SetInsertPoint(Phi->getParent()->getFirstNonPHI());

  llvm::Instruction *LoadedValue = Builder.CreateLoad(Alloca->getAllocatedType(), Alloca);
  Phi->replaceAllUsesWith(LoadedValue);

  Phi->eraseFromParent();

  return LoadedValue;
}

bool demotePHIsToAllocas(llvm::Function &F) {
  std::vector<llvm::PHINode *> PHIs;

  for (auto &BB : F)
    for (auto &I : BB)
      if (auto *PHI = llvm::dyn_cast<llvm::PHINode>(&I))
        PHIs.push_back(PHI);

  bool Changed = false;
  HIPSYCL_DEBUG_INFO << "Break PHIs to alloca:\n";
  for (auto *I : PHIs) {
    HIPSYCL_DEBUG_INFO << "  ";
    HIPSYCL_DEBUG_EXECUTE_INFO(I->print(llvm::outs()); llvm::outs() << "\n";)
    breakPHIToAllocas(I);
    Changed = true;
  }
  return Changed;
}

} // namespace

namespace hipsycl::compiler {

char PHIsToAllocasPassLegacy::ID = 0;

void PHIsToAllocasPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();

  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::PostDominatorTreeWrapperPass>();
}

bool PHIsToAllocasPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;

  return demotePHIsToAllocas(F);
}

llvm::PreservedAnalyses PHIsToAllocasPass::run(llvm::Function &F,
                                               llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA =
      MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA)) {
    return llvm::PreservedAnalyses::all();
  }

  if (!demotePHIsToAllocas(F))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
