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

#include "hipSYCL/compiler/PHIsToAllocas.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

//#include "Workgroup.h"
//#include "WorkitemHandlerChooser.h"
//#include "WorkitemLoops.h"

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
llvm::Instruction *breakPHIToAllocas(llvm::PHINode *Phi, hipsycl::compiler::VariableUniformityInfo &VUA) {

  // Loop iteration variables can be detected only when they are
  // implemented using PHI nodes. Maintain information of the
  // split PHI nodes in the VUA by first analyzing the function
  // with the PHIs intact and propagating the uniformity info
  // of the PHI nodes.
  std::string AllocaName{std::string(Phi->getName().str()) + ".ex_phi"};

  llvm::Function *Function = Phi->getParent()->getParent();

  const bool OriginalPHIWasUniform = VUA.isUniform(Function, Phi);

  llvm::IRBuilder Builder(&*(Function->getEntryBlock().getFirstInsertionPt()));

  llvm::Instruction *Alloca = Builder.CreateAlloca(Phi->getType(), 0, AllocaName);

  for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues(); ++Incoming) {
    auto *V = Phi->getIncomingValue(Incoming);
    auto *IncomingBB = Phi->getIncomingBlock(Incoming);
    Builder.SetInsertPoint(IncomingBB->getTerminator());
    llvm::Instruction *Store = Builder.CreateStore(V, Alloca);
    if (OriginalPHIWasUniform)
      VUA.setUniform(Function, Store);
  }
  Builder.SetInsertPoint(Phi);

  llvm::Instruction *LoadedValue = Builder.CreateLoad(Alloca);
  Phi->replaceAllUsesWith(LoadedValue);

  if (OriginalPHIWasUniform) {
#ifdef DEBUG_PHIS_TO_ALLOCAS
    std::cout << "PHIsToAllocas: Original PHI was uniform" << std::endl << "original:";
    phi->dump();
    std::cout << "Alloca:";
    Alloca->dump();
    std::cout << "loadedValue:";
    loadedValue->dump();
#endif
    VUA.setUniform(Function, Alloca);
    VUA.setUniform(Function, LoadedValue);
  }
  Phi->eraseFromParent();

  return LoadedValue;
}

bool demotePHIsToAllocas(llvm::Function &F, hipsycl::compiler::VariableUniformityInfo &VUA, const llvm::LoopInfo &LI) {
  std::vector<llvm::PHINode *> PHIs;

  auto BBsInWI = hipsycl::compiler::utils::getBasicBlocksInWorkItemLoops(LI);

  for (auto *BB : BBsInWI) {
    for (auto &I : *BB) {
      if (llvm::isa<llvm::PHINode>(I)) {
        PHIs.push_back(llvm::cast<llvm::PHINode>(&I));
      }
    }
  }

  bool Changed = false;
  HIPSYCL_DEBUG_INFO << "Break PHIs to alloca:\n";
  for (auto *I : PHIs) {
    HIPSYCL_DEBUG_INFO << "  ";
    HIPSYCL_DEBUG_EXECUTE_INFO(I->print(llvm::outs()); llvm::outs()  << "\n";)
    breakPHIToAllocas(I, VUA);
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

  AU.addRequired<VariableUniformityAnalysisLegacy>();
  AU.addPreserved<VariableUniformityAnalysisLegacy>();

  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::PostDominatorTreeWrapperPass>();
}

bool PHIsToAllocasPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;

  const auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  auto &VUA = getAnalysis<VariableUniformityAnalysisLegacy>().getResult();

  return demotePHIsToAllocas(F, VUA, LI);
}

llvm::PreservedAnalyses PHIsToAllocasPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA)) {
    return llvm::PreservedAnalyses::all();
  }

  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);
  const auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (!demotePHIsToAllocas(F, VUA, LI))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
