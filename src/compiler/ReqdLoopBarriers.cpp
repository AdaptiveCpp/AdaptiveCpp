// LLVM loop pass that adds required barriers to loops.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2019 Pekka Jääskeläinen
//               2021 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/ReqdLoopBarriers.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <iostream>

//#define DEBUG_LOOP_BARRIERS

namespace {
bool addRequiredBarriersToLoop(llvm::Loop *L, llvm::DominatorTree &DT,
                               hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  if (!hipsycl::compiler::utils::hasBarriers(*L->getHeader()->getParent(), SAA))
    return false;

  if (!hipsycl::compiler::utils::isInWorkItemLoop(*L))
    return false;

  bool isBLoop = false;
  bool changed = false;

  for (auto BBIt = L->block_begin(), BBEIt = L->block_end(); BBIt != BBEIt && !isBLoop; ++BBIt) {
    for (auto &I : **BBIt) {
      if (hipsycl::compiler::utils::isBarrier(&I, SAA)) {
        isBLoop = true;
        break;
      }
    }
  }

  for (auto BBIt = L->block_begin(), BBEIt = L->block_end(); BBIt != BBEIt && isBLoop; ++BBIt) {
    for (auto &I : **BBIt) {
      if (hipsycl::compiler::utils::isBarrier(&I, SAA)) {

        // Found a barrier in this loop:
        // 1) add a barrier in the loop header.
        // 2) add a barrier in the latches

        // Add a barrier on the preheader to ensure all WIs reach
        // the loop header with all the previous code already
        // executed.
        llvm::BasicBlock *PreHeader = L->getLoopPreheader();
        assert((PreHeader != NULL) && "Non-canonicalized loop found!\n");
#ifdef DEBUG_LOOP_BARRIERS
        std::cerr << "### adding to preheader BB" << std::endl;
        preheader->dump();
        std::cerr << "### before instr" << std::endl;
        preheader->getTerminator()->dump();
#endif
        hipsycl::compiler::utils::createBarrier(PreHeader->getTerminator(), SAA);
        PreHeader->setName(PreHeader->getName() + ".loopbarrier");

        // Add a barrier after the PHI nodes on the header (the replicated
        // headers will be merged afterwards).
        if (llvm::BasicBlock *Header = L->getHeader(); Header->getFirstNonPHI() != &Header->front()) {
          hipsycl::compiler::utils::createBarrier(Header->getFirstNonPHI(), SAA);
          Header->setName(Header->getName() + ".phibarrier");
          // Split the block to  create a replicable region of
          // the loop contents in case the phi node contains a
          // branch (that can be to inside the region).
          //          if (Header->getTerminator()->getNumSuccessors() > 1)
          //    SplitBlock(Header, Header->getTerminator(), this);
        }

        // Add the barriers on the exiting block and the latches,
        // which might not always be the same if there is computation
        // after the exit decision.
        llvm::BasicBlock *BrExit = L->getExitingBlock();
        if (BrExit) {
          hipsycl::compiler::utils::createBarrier(BrExit->getTerminator(), SAA);
          BrExit->setName(BrExit->getName() + ".brexitbarrier");
        }

        if (llvm::BasicBlock *Latch = L->getLoopLatch(); BrExit != Latch) {
          // This loop has only one latch. Do not check for dominance, we
          // are probably running before BTR.
          hipsycl::compiler::utils::createBarrier(Latch->getTerminator(), SAA);
          Latch->setName(Latch->getName() + ".latchbarrier");
          return changed;
        }

        llvm::SmallVector<llvm::BasicBlock *, 4> Latches;
        L->getLoopLatches(Latches);
        for (auto *Latch : Latches) {
          // Latch found in the loop, see if the barrier dominates it
          // (otherwise if might no even belong to this "tail", see
          // forifbarrier1 graph test).
          if (DT.dominates(*BBIt, Latch)) {
            hipsycl::compiler::utils::createBarrier(Latch->getTerminator(), SAA);
            Latch->setName(Latch->getName() + ".latchbarrier");
          }
        }
        return true;
      }
    }
  }

  /* This is a loop without a barrier. Ensure we have a non-barrier
     block as a preheader so we can replicate the loop as a whole.

     If the block has proper instructions after the barrier, it
     will be split in CanonicalizeBarriers. */
  llvm::BasicBlock *PreHeader = L->getLoopPreheader();
  assert((PreHeader != NULL) && "Non-canonicalized loop found!\n");

  llvm::Instruction *t = PreHeader->getTerminator();
  llvm::Instruction *Prev = NULL;
  if (&PreHeader->front() != t)
    Prev = t->getPrevNode();
  if (Prev && hipsycl::compiler::utils::isBarrier(Prev, SAA)) {
    llvm::BasicBlock *new_b = SplitBlock(PreHeader, t);
    new_b->setName(PreHeader->getName() + ".postbarrier_dummy");
    return true;
  }

  return changed;
}
} // namespace

namespace hipsycl::compiler {
char AddRequiredLoopBarriersPassLegacy::ID = 0;

void AddRequiredLoopBarriersPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<VariableUniformityAnalysisLegacy>();
}

bool AddRequiredLoopBarriersPassLegacy::runOnLoop(llvm::Loop *L, llvm::LPPassManager &LPM) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(L->getHeader()->getParent()) || !utils::hasBarriers(*L->getHeader()->getParent(), SAA))
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();

  return addRequiredBarriersToLoop(L, DT, SAA);
}

llvm::PreservedAnalyses AddRequiredLoopBarriersPass::run(llvm::Loop &L, llvm::LoopAnalysisManager &AM,
                                                         llvm::LoopStandardAnalysisResults &AR,
                                                         llvm::LPMUpdater &LPMU) {
  auto *F = L.getHeader()->getParent();
  auto &FAM = AM.getResult<llvm::FunctionAnalysisManagerLoopProxy>(L, AR);

  auto *MAM = FAM.getCachedResult<llvm::ModuleAnalysisManagerFunctionProxy>(*F);
  auto *SAA = MAM->getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F->getParent());
  if (!SAA || !SAA->isKernelFunc(F) || !utils::hasBarriers(*F, *SAA))
    return llvm::PreservedAnalyses::all();

  if (!addRequiredBarriersToLoop(&L, AR.DT, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA = llvm::getLoopPassPreservedAnalyses();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
