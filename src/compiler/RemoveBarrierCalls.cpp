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

#include "hipSYCL/compiler/RemoveBarrierCalls.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

namespace {
bool removeBarrierCalls(llvm::Function &F, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  if (!SAA.isKernelFunc(&F))
    return false;

  // Collect the barrier calls to be removed first, not remove them
  // instantly as it'd invalidate the iterators.
  llvm::SmallPtrSet<llvm::Instruction *, 8> BarriersToRemove;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (hipsycl::compiler::utils::isBarrier(&I, SAA)) {
        BarriersToRemove.insert(&I);
      }
    }
  }

  for (auto *B : BarriersToRemove) {
    HIPSYCL_DEBUG_INFO << "Remove barrier ";
    HIPSYCL_DEBUG_EXECUTE_INFO(B->print(llvm::outs()); llvm::outs() << " from " << B->getParent()->getName() << "\n";)
    B->eraseFromParent();
  }
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  auto *M = F.getParent();
  if (auto *B = M->getFunction(hipsycl::compiler::BarrierIntrinsicName)) {
    if (B->getNumUses() == 0) {
      B->eraseFromParent();
      HIPSYCL_DEBUG_INFO << "Clean-up helper barrier: " << hipsycl::compiler::BarrierIntrinsicName << "\n";
    }
  }
  return !BarriersToRemove.empty();
}

} // namespace

namespace hipsycl::compiler {

char RemoveBarrierCallsPassLegacy::ID = 0;

bool RemoveBarrierCallsPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  return removeBarrierCalls(F, SAA);
}

void RemoveBarrierCallsPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addPreserved<VariableUniformityAnalysisLegacy>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

llvm::PreservedAnalyses RemoveBarrierCallsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA)
    return llvm::PreservedAnalyses::all();

  if (!removeBarrierCalls(F, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
