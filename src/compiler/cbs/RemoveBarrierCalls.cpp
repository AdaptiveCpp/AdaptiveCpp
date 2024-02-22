// LLVM function pass to remove all barrier calls.
//
// Copyright (c) 2016 Pekka Jääskeläinen / TUT
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

#include "hipSYCL/compiler/cbs/RemoveBarrierCalls.hpp"

#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include <llvm/Support/Casting.h>

namespace hipsycl::compiler {
namespace {

using namespace cbs;

bool deleteGlobalVariable(llvm::Module *M, llvm::StringRef VarName) {
  if (auto *GV = M->getGlobalVariable(VarName)) {
    llvm::SmallVector<llvm::Instruction *, 8> WL;
    for (auto U : GV->users())
      if (auto LI = llvm::dyn_cast<llvm::LoadInst>(U); LI && LI->user_empty())
        WL.push_back(LI);
    for (auto *LI : WL)
      LI->eraseFromParent();

    if (GV->getNumUses() == 0 ||
        std::none_of(GV->user_begin(), GV->user_end(), [GV](llvm::User *U) { return U != GV; })) {
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Clean-up global variable " << *GV << "\n";
      GV->eraseFromParent();
      return true;
    }
    HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Global variable still in use " << VarName << "\n";
    for (auto *U : GV->users()) {
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] >>> " << *U;
      if (auto I = llvm::dyn_cast<llvm::Instruction>(U)) {
        HIPSYCL_DEBUG_EXECUTE_INFO(
          llvm::outs() << " in " << I->getFunction()->getName()
        );
      }
      HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << "\n");
    }
  }
  return false;
}

bool removeBarrierCalls(llvm::Function &F, SplitterAnnotationInfo &SAA) {
  if (!SAA.isKernelFunc(&F))
    return false;

  // Collect the barrier calls to be removed first, not remove them
  // instantly as it'd invalidate the iterators.
  llvm::SmallPtrSet<llvm::Instruction *, 8> BarriersToRemove;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (utils::isBarrier(&I, SAA)) {
        BarriersToRemove.insert(&I);
      }
    }
  }

  for (auto *B : BarriersToRemove) {
    HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Remove barrier ";
    HIPSYCL_DEBUG_EXECUTE_INFO(B->print(llvm::outs());
                               llvm::outs() << " from " << B->getParent()->getName() << "\n";)
    B->eraseFromParent();
  }

  auto *M = F.getParent();
  if (auto *B = M->getFunction(BarrierIntrinsicName)) {
    if (B->getNumUses() == 0) {
      B->eraseFromParent();
      SAA.removeSplitter(B);
      HIPSYCL_DEBUG_INFO << "[RemoveBarrierCalls] Clean-up helper barrier: "
                         << BarrierIntrinsicName << "\n";
    }
  }

  bool Changed = !BarriersToRemove.empty();

  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameX);
  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameY);
  Changed |= deleteGlobalVariable(M, LocalIdGlobalNameZ);

  return Changed;
}

} // namespace


char RemoveBarrierCallsPassLegacy::ID = 0;

bool RemoveBarrierCallsPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  return removeBarrierCalls(F, SAA);
}

void RemoveBarrierCallsPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses RemoveBarrierCallsPass::run(llvm::Function &F,
                                                    llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA)
    return llvm::PreservedAnalyses::all();

  if (!removeBarrierCalls(F, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
