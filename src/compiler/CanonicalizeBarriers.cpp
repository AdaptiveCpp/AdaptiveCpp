// LLVM function pass to canonicalize barriers.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos
//               2012-2014 Pekka Jääskeläinen / Tampere University of Technology
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

#include "hipSYCL/compiler/CanonicalizeBarriers.hpp"
#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <iostream>

namespace {

// Canonicalize barriers: ensure all barriers are in a separate BB
// containing only the barrier and the terminator, with just one
// predecessor. This allows us to use those BBs as markers only,
// they will not be replicated.
bool canonicalizeBarriers(llvm::Function &F, const llvm::LoopInfo &LI,
                          const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  using namespace hipsycl::compiler;
  bool Changed = false;

  auto *WILoop = utils::getSingleWorkItemLoop(LI);
  assert(WILoop && "No WI Loop found!");

  llvm::BasicBlock *Entry = utils::getWorkItemLoopBodyEntry(WILoop);
  assert(Entry && "No WI Loop Entry found!");

  if (!utils::hasOnlyBarrier(Entry, SAA)) {
    llvm::BasicBlock *EffectiveEntry = SplitBlock(Entry, &(Entry->front()));

    EffectiveEntry->takeName(Entry);
    Entry->setName("entry.barrier");
    utils::createBarrier(Entry->getTerminator(), SAA);
    Changed = true;
  }

  auto *WILatch = WILoop->getLoopLatch();
  assert(WILatch && "No WI Latch found!");
  llvm::SmallVector<llvm::BasicBlock *, 4> Exits{llvm::pred_begin(WILatch), llvm::pred_end(WILatch)};

  for (auto *BB : Exits) {
    auto *T = BB->getTerminator();

    // If conditional branch, split the edge, so we have a barrier only block.
    if (T->getNumSuccessors() > 1) {
      BB = utils::splitEdge(BB, WILatch, nullptr, nullptr);
      T = BB->getTerminator();
      BB->setName("exit.barrier");
      utils::createBarrier(T, SAA);
      Changed = true;
      continue;
    }

    // The function exits should have barriers.
    if (!utils::hasOnlyBarrier(BB, SAA)) {
      /* In case the bb is already terminated with a barrier,
         split before the barrier so we don'T create an empty
         parallel region.

         This is because the assumptions of the other passes in the
         compilation that are
         a) exit node is a barrier block
         b) there are no empty parallel regions (which would be formed
         between the explicit barrier and the added one). */
      llvm::BasicBlock *Exit;
      if (utils::endsWithBarrier(BB, SAA))
        Exit = SplitBlock(BB, T->getPrevNode());
      else
        Exit = SplitBlock(BB, T);
      Exit->setName("exit.barrier");
      utils::createBarrier(T, SAA);
      Changed = true;
    }
  }

  llvm::SmallPtrSet<llvm::Instruction *, 8> Barriers;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (hipsycl::compiler::utils::isBarrier(&I, SAA)) {
        Barriers.insert(&I);
      }
    }
  }

  // Finally add all the split points, now that we are done with the
  // iterators.
  for (auto *Barrier : Barriers) {
    llvm::BasicBlock *BB = Barrier->getParent();
    HIPSYCL_DEBUG_INFO << "[Canonicalize] Barrier in: " << BB->getName() << "\n";

    // Split post barrier first cause it does not make the barrier
    // to belong to another basic block.
    llvm::Instruction *T = BB->getTerminator();
    // if ((T->getNumSuccessors() > 1) ||
    //     (T->getPrevNode() != *i)) {
    // Change: barriers with several successors are all right
    // they just start several parallel regions. Simplifies
    // loop handling.

    const bool HasNonBranchInstructionsAfterBarrier = T->getPrevNode() != Barrier;

    if (HasNonBranchInstructionsAfterBarrier) {
      HIPSYCL_DEBUG_INFO << "[Canonicalize] Splitting after barrier in: " << BB->getName() << "\n";
      llvm::BasicBlock *NewB = SplitBlock(BB, Barrier->getNextNode());
      NewB->setName(BB->getName() + ".postbarrier");
      Changed = true;
    }

    llvm::BasicBlock *Predecessor = BB->getSinglePredecessor();
    if (Predecessor != NULL) {
      auto *Pt = Predecessor->getTerminator();
      if ((Pt->getNumSuccessors() == 1) && (&BB->front() == Barrier)) {
        // Barrier is at the beginning of the BB,
        // which has a single predecessor with just
        // one successor (the barrier itself), thus
        // no need to split before barrier.
        continue;
      }
    }
    if ((BB == &(BB->getParent()->getEntryBlock())) && (&BB->front() == Barrier))
      continue;

    // If no instructions before barrier, do not split
    // (allow multiple predecessors, eases loop handling).
    // if (&BB->front() == (*i))
    //   continue;
    HIPSYCL_DEBUG_INFO << "[Canonicalize] Splitting before barrier in: " << BB->getName() << "\n";

    llvm::BasicBlock *NewB = SplitBlock(BB, Barrier);
    NewB->takeName(BB);
    BB->setName(NewB->getName() + ".prebarrier");
    Changed = true;
  }

  // Prune empty regions. That is, if there are two successive
  // pure barrier blocks without side branches, remove the other one.
  bool EmptyRegionDeleted;
  do {
    EmptyRegionDeleted = false;
    for (auto &BB : F) {
      auto *T = BB.getTerminator();
      if (!hipsycl::compiler::utils::endsWithBarrier(&BB, SAA) || T->getNumSuccessors() != 1)
        continue;

      llvm::BasicBlock *Successor = T->getSuccessor(0);

      if (hipsycl::compiler::utils::hasOnlyBarrier(Successor, SAA) && Successor->getSinglePredecessor() == &BB) {
        BB.replaceAllUsesWith(Successor);
        BB.eraseFromParent();
        EmptyRegionDeleted = true;
        Changed = true;
        break;
      }
    }
  } while (EmptyRegionDeleted);

  return Changed;
}

} // namespace

namespace hipsycl::compiler {
char CanonicalizeBarriersPassLegacy::ID = 0;

void CanonicalizeBarriersPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();

  AU.addPreserved<VariableUniformityAnalysisLegacy>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool CanonicalizeBarriersPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;
  const auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  return canonicalizeBarriers(F, LI, SAA);
}

llvm::PreservedAnalyses CanonicalizeBarriersPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F)) {
    return llvm::PreservedAnalyses::all();
  }

  const auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (!canonicalizeBarriers(F, LI, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler