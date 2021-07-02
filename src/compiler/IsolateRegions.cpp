// Implementation of IsolateRegions RegionPass.
//
// Copyright (c) 2012-2015 Pekka Jääskeläinen / TUT
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

#include "hipSYCL/compiler/IsolateRegions.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

#include "llvm/Analysis/DominanceFrontier.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <iostream>

//#define DEBUG_ISOLATE_REGIONS

/* Ensure Single-Entry Single-Exit Regions are isolated from the
   exit node so they won't get split illegally with tail replication.

   This might happen in case an if .. else .. structure is just
   before an exit from kernel. Both branches are split even though
   we would like to replicate the structure as a whole to retain
   semantics. This adds dummy basic blocks to all Regions just for
   clarity. Cleanup with -simplifycfg.

   TODO: Also add a dummy BB in case the Region starts with a
   barrier. Such a Region might not get optimally replicated and
   can lead to problematic cases. E.g.:

   digraph G {
      BAR1 -> A;
      A -> X;
      BAR1 -> X;
      X -> BAR2;
   }

   (draw with "dot -Tpng -o graph.png"   + copy paste the above)

   Here you have a structure which should be replicated fully but
   it won't as the Region starts with a barrier at a split point
   BB, thus it tries to replicate both of the branches which lead
   to interesting errors and is not supported. Another option would
   be to tail replicate both of the branches, but currently tail
   replication is done only starting from the exit nodes.

   IsolateRegions "normalizes" the graph to:

   digraph G {
      BAR1 -> r_entry;
      r_entry -> A;
      A -> X;
      r_entry -> X;
      X -> BAR2;
   }
*/

namespace {
using namespace hipsycl::compiler;

/**
 * Adds a dummy node after the given basic block.
 */
void addDummyAfter(llvm::Region *R, llvm::BasicBlock *BB) {
  llvm::BasicBlock *NewEntry = SplitBlock(BB, BB->getTerminator());
  NewEntry->setName(BB->getName() + ".r_entry");
  R->replaceEntry(NewEntry);
}

/**
 * Adds a dummy node before the given basic block.
 *
 * The edges going in to the original BB are moved to go
 * in to the dummy BB in case the source BB is inside the
 * same region.
 */
void addDummyBefore(llvm::Region *R, llvm::BasicBlock *BB) {
  std::vector<llvm::BasicBlock *> RegionPreds;

  for (auto *Pred : llvm::predecessors(BB)) {
    if (R->contains(Pred))
      RegionPreds.push_back(Pred);
  }
  llvm::BasicBlock *NewExit = SplitBlockPredecessors(BB, RegionPreds, ".r_exit");
  R->replaceExit(NewExit);
}

bool isolateRegion(llvm::Region *R, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  llvm::BasicBlock *Exit = R->getExit();
  if (!Exit)
    return false;

  HIPSYCL_DEBUG_INFO << "Isolating Region: " << R->getEntry()->getName() << "\n";

  const bool IsFunctionExit = Exit->getTerminator()->getNumSuccessors() == 0;

  bool Changed = false;

  if (utils::blockHasBarrier(Exit, SAA) || IsFunctionExit) {
    addDummyBefore(R, Exit);
    Changed = true;
  }

  llvm::BasicBlock *Entry = R->getEntry();
  if (!Entry)
    return Changed;

  // todo: wi header?
  bool IsFunctionEntry = &Entry->getParent()->getEntryBlock() == Entry;

  if (utils::blockHasBarrier(Entry, SAA) || IsFunctionEntry) {
    addDummyAfter(R, Entry);
    Changed = true;
  }

  return Changed;
}

bool isolateRegions(const hipsycl::compiler::SplitterAnnotationInfo &SAA, const llvm::LoopInfo &LI,
                    const llvm::RegionInfo &RI) {
  llvm::SmallVector<llvm::Region *, 8> WorkList{RI.getTopLevelRegion()};
  bool Changed = false;

  do {
    llvm::SmallVector<llvm::Region *, 8> CurrentRegions;
    for (auto *R : WorkList) {
      if (utils::isInWorkItemLoop(*R, LI))
        Changed |= isolateRegion(R, SAA);
      std::transform(R->begin(), R->end(), std::back_inserter(CurrentRegions), [](auto &UR) { return UR.get(); });
    }

    WorkList.swap(CurrentRegions);
  } while (!WorkList.empty());
  return Changed;
}

} // namespace

namespace hipsycl::compiler {
char IsolateRegionsPassLegacy::ID = 0;

void IsolateRegionsPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addPreserved<VariableUniformityAnalysisLegacy>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();

  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::PostDominatorTreeWrapperPass>();
}
bool IsolateRegionsPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;

  const auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  llvm::DominanceFrontier DF{};
  DF.analyze(DT);
  llvm::RegionInfo RI;
  RI.recalculate(F, &DT, &PDT, &DF);
  return isolateRegions(SAA, LI, RI);
}

llvm::PreservedAnalyses IsolateRegionsPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());

  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA)) {
    return llvm::PreservedAnalyses::all();
  }

  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  auto &RI = AM.getResult<llvm::RegionInfoAnalysis>(F);
  isolateRegions(*SAA, LI, RI);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::RegionInfoAnalysis>();
  return PA;
}

} // namespace hipsycl::compiler