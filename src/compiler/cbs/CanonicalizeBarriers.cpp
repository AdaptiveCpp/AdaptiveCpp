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

#include "hipSYCL/compiler/cbs/CanonicalizeBarriers.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/PipelineBuilder.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"

#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <iostream>

namespace {
using namespace hipsycl::compiler;

bool canonicalizeExitBarriers(llvm::Function &F, SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  llvm::SmallVector<llvm::BasicBlock *, 4> Exits;
  for (auto &BB : F)
    if (BB.getTerminator()->getNumSuccessors() == 0)
      Exits.push_back(&BB);

  for (auto *BB : Exits) {
    auto *T = BB->getTerminator();

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
  return Changed;
}

bool pruneEmptyRegions(llvm::Function &F, const SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  // Prune empty regions. That is, if there are two successive
  // pure barrier blocks without side branches, remove the other one.
  bool EmptyRegionDeleted;
  do {
    EmptyRegionDeleted = false;
    for (auto &BB : F) {
      auto *T = BB.getTerminator();
      if (!utils::hasOnlyBarrier(&BB, SAA) || T->getNumSuccessors() != 1)
        continue;

      llvm::BasicBlock *Successor = T->getSuccessor(0);

      if (utils::hasOnlyBarrier(Successor, SAA) && Successor->getSinglePredecessor() == &BB) {
        HIPSYCL_DEBUG_INFO << "Prune BasicBlock: " << BB.getName() << "\n";
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

bool canonicalizeEntry(llvm::BasicBlock *Entry, SplitterAnnotationInfo &SAA) {
  bool Changed = false;
  if (!utils::hasOnlyBarrier(Entry, SAA)) {
    llvm::BasicBlock *EffectiveEntry = SplitBlock(Entry, &(Entry->front()));

    EffectiveEntry->takeName(Entry);
    Entry->setName("entry.barrier");
    utils::createBarrier(Entry->getTerminator(), SAA);
    Changed = true;
  }
  return Changed;
}

// Canonicalize barriers: ensure all barriers are in a separate BB
// containing only the barrier and the terminator, with just one
// predecessor. This allows us to use those BBs as markers only,
// they will not be replicated.
bool canonicalizeBarriers(llvm::Function &F, SplitterAnnotationInfo &SAA) {
  bool Changed = false;

  llvm::BasicBlock *Entry = &F.getEntryBlock();

  Changed |= canonicalizeEntry(Entry, SAA);
  Changed |= canonicalizeExitBarriers(F, SAA);

  llvm::SmallPtrSet<llvm::Instruction *, 8> Barriers;

  for (auto &BB : F)
    for (auto &I : BB)
      if (utils::isBarrier(&I, SAA))
        Barriers.insert(&I);

  // Finally add all the split points, now that we are done with the
  // iterators.
  for (auto *Barrier : Barriers) {
    llvm::BasicBlock *BB = Barrier->getParent();
    HIPSYCL_DEBUG_INFO << "[Canonicalize] Barrier in: " << BB->getName() << "\n";

    // Split post barrier first cause it does not make the barrier
    // to belong to another basic block.
    llvm::Instruction *T = BB->getTerminator();

    // looses conditional branches if in the same BB as a barrier, must split if multiple successors
    if (T->getPrevNode() != Barrier || T->getNumSuccessors() > 1) {
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
    if (BB == Entry && (&BB->front() == Barrier))
      continue;

    HIPSYCL_DEBUG_INFO << "[Canonicalize] Splitting before barrier in: " << BB->getName() << "\n";

    llvm::BasicBlock *NewB = SplitBlock(BB, Barrier);
    NewB->takeName(BB);
    BB->setName(NewB->getName() + ".prebarrier");
    Changed = true;
  }

  Changed |= pruneEmptyRegions(F, SAA);

  return Changed;
}

} // namespace

namespace hipsycl::compiler {
char CanonicalizeBarriersPassLegacy::ID = 0;

void CanonicalizeBarriersPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool CanonicalizeBarriersPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;
  return canonicalizeBarriers(F, SAA);
}

llvm::PreservedAnalyses CanonicalizeBarriersPass::run(llvm::Function &F,
                                                      llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA))
    return llvm::PreservedAnalyses::all();

  if (!canonicalizeBarriers(F, *SAA))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
