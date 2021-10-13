// LLVM function pass to create loops that run all the work items
// in a work group while respecting barrier synchronization points.
//
// Copyright (c) 2012-2019 Pekka Jääskeläinen
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

#define DEBUG_TYPE "workitem-loops"

#include "hipSYCL/compiler/WorkItemLoopCreation.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/ParallelRegion.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <map>
#include <vector>

//#define DUMP_CFGS

#define DEBUG_WORK_ITEM_LOOPS
#define DEBUG_PR_CREATION
#define DEBUG_REFERENCE_FIXING

#define CONTEXT_ARRAY_ALIGN 64

namespace {
using namespace hipsycl::compiler;

template <class Predicate>
void addPredecessorsIf(llvm::SmallVectorImpl<llvm::BasicBlock *> &Preds, llvm::BasicBlock *BB, Predicate &&P) {
  for (auto *Pred : llvm::predecessors(BB)) {
    if (P(Pred))
      Preds.push_back(Pred);
  }
}

bool verifyNoBarriers(const llvm::BasicBlock *B, const SplitterAnnotationInfo &SAA) {
  for (auto &I : *B) {
    if (utils::isBarrier(&I, SAA))
      return false;
  }

  return true;
}

class WorkItemLoopCreator {

public:
  static char ID;

  WorkItemLoopCreator(llvm::Function &F, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT, llvm::LoopInfo &LI,
                      VariableUniformityInfo &VUA, SplitterAnnotationInfo &SAA);

private:
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::set<llvm::Instruction *> InstructionIndex;
  typedef std::vector<llvm::Instruction *> InstructionVec;
  typedef std::map<std::string, llvm::Instruction *> StrInstructionMap;

  llvm::DominatorTree &DT;
  llvm::LoopInfo &LI;
  llvm::PostDominatorTree &PDT;
  SplitterAnnotationInfo &SAA;
  VariableUniformityInfo &VUA;

  ParallelRegion::ParallelRegionVector *OriginalParallelRegions;

  llvm::Type *SizeT;
  StrInstructionMap ContextArrays;
  llvm::BasicBlock *WILoopEntry;
  llvm::BasicBlock *WILoopExit;
  llvm::Loop *WorkItemLoop;

  std::size_t ContextArraySize;

public:
  bool processFunction(llvm::Function &F);

  bool fixUndominatedVariableUses(llvm::Function &F);

private:
  void fixMultiRegionVariables(ParallelRegion *Region);
  void fixMultiRegionAllocas(llvm::Function *F);
  void addContextSaveRestore(llvm::Instruction *I);
  void releaseParallelRegions();
  ParallelRegion::ParallelRegionVector *getParallelRegions(llvm::Loop &L);
  void getExitBlocks(llvm::Function &F, llvm::SmallVectorImpl<llvm::BasicBlock *> &ExitBlocks);
  ParallelRegion *createParallelRegionBefore(llvm::BasicBlock *B);

  llvm::Instruction *addContextSave(llvm::Instruction *I, llvm::Instruction *Alloca);
  llvm::Instruction *addContextRestore(llvm::Value *Val, llvm::Instruction *Alloca, bool PoclWrapperStructAdded,
                                       llvm::Instruction *Before = NULL, bool IsAlloca = false);
  llvm::Instruction *getContextArray(llvm::Instruction *Instruction, bool &PoclWrapperStructAdded);

  std::pair<llvm::BasicBlock *, llvm::BasicBlock *> createLoopAround(ParallelRegion &Region, llvm::BasicBlock *EntryBb,
                                                                     llvm::BasicBlock *ExitBb, bool PeeledFirst);

  ParallelRegion *regionOfBlock(llvm::BasicBlock *BB);

  bool shouldNotBeContextSaved(llvm::Instruction *Instr);

  std::map<llvm::Instruction *, unsigned> TempInstructionIds;
  size_t TempInstructionIndex;
  // An alloca in the kernel which stores the first iteration to execute
  // in the inner (dimension 0) loop. This is set to 1 in an peeled iteration
  // to skip the 0, 0, 0 iteration in the loops.
  //  llvm::Value *LocalIdXFirstVar;
  void removeOriginalWILoop();
};

std::pair<llvm::BasicBlock *, llvm::BasicBlock *> WorkItemLoopCreator::createLoopAround(ParallelRegion &Region,
                                                                                        llvm::BasicBlock *EntryBb,
                                                                                        llvm::BasicBlock *ExitBb,
                                                                                        bool PeeledFirst) {
  /*

    Generate a structure like this for each loop level (x,y,z):

    for.init:

    ; if peeledFirst is false:
    store i32 0, i32* %_local_id_x, align 4

    ; if peeledFirst is true (assume the 0,0,0 iteration has been executed earlier)
    ; assume _local_id_x_first is is initialized to 1 in the peeled pregion copy
    store _local_id_x_first, i32* %_local_id_x, align 4
    store i32 0, %_local_id_x_first

    br label %for.body

    for.body:

    ; the parallel region code here

    br label %for.inc

    for.inc:

    ; Separated inc and cond check blocks for easier loop unrolling later on.
    ; Can then chain N times for.body+for.inc to unroll.

    %2 = load i32* %_local_id_x, align 4
    %inc = add nsw i32 %2, 1

    store i32 %inc, i32* %_local_id_x, align 4
    br label %for.cond

    for.cond:

    ; loop header, compare the id to the local size
    %0 = load i32* %_local_id_x, align 4
    %cmp = icmp ult i32 %0, i32 123
    br i1 %cmp, label %for.body, label %for.end

    for.end:

    OPTIMIZE: Use a separate iteration variable across all the loops to iterate the context
    data arrays to avoid needing multiplications to find the correct location, and to
    enable easy vectorization of loading the context data when there are parallel iterations.
  */

  llvm::BasicBlock *LoopBodyEntryBb = EntryBb;
  llvm::LLVMContext &C = LoopBodyEntryBb->getContext();
  llvm::Function *F = LoopBodyEntryBb->getParent();
  LoopBodyEntryBb->setName(std::string("pregion_for_entry.") + EntryBb->getName().str());

  assert(ExitBb->getTerminator()->getNumSuccessors() == 1);

  llvm::BasicBlock *OldExit = ExitBb->getTerminator()->getSuccessor(0);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(llvm::errs() << "cfg for OldExit: " << OldExit->getName() << "\n"; F->viewCFG();)
  llvm::ValueToValueMapTy VMap;
  auto *WIHeader = WorkItemLoop->getHeader();
  auto *WIPreHeader = utils::splitEdge(WorkItemLoop->getLoopPreheader(), WIHeader, &LI, &DT);
  auto *WILatch = WorkItemLoop->getLoopLatch();
  auto *PreHeader = llvm::CloneBasicBlock(WIPreHeader, VMap, ".pregion_for_init", F, nullptr, nullptr);
  auto *Header = llvm::CloneBasicBlock(WIHeader, VMap, ".pregion_for_cond", F, nullptr, nullptr);
  auto *Latch = llvm::CloneBasicBlock(WILatch, VMap, ".pregion_for_inc", F, nullptr, nullptr);

  if (WIPreHeader) {
    HIPSYCL_DEBUG_INFO << "PreHeader: " << WIPreHeader->getName() << " clone: " << PreHeader->getName() << "\n";
  }
  if (WIHeader) {
    HIPSYCL_DEBUG_INFO << "Header: " << WIHeader->getName() << " clone: " << Header->getName() << "\n";
  }
  if (WILatch) {
    HIPSYCL_DEBUG_INFO << "Latch: " << WILatch->getName() << " clone: " << Latch->getName() << "\n";
  }
  llvm::outs().flush();

  HIPSYCL_DEBUG_INFO << "loop header:";
  HIPSYCL_DEBUG_EXECUTE_INFO(Header->print(llvm::outs()); WIHeader->print(llvm::outs());)

  VMap[WIPreHeader] = PreHeader;
  VMap[WILatch] = Latch;
  Header->replacePhiUsesWith(WIPreHeader, PreHeader);
  Header->replacePhiUsesWith(WILatch, Latch);
  Header->getTerminator()->setSuccessor(0, LoopBodyEntryBb);
  Header->getTerminator()->setSuccessor(1, OldExit);
  Latch->getTerminator()->setSuccessor(0, Header);

  llvm::SmallVector<llvm::BasicBlock *, 4> NewBlocks{PreHeader, Latch, Header};
  llvm::remapInstructionsInBlocks(NewBlocks, VMap);
  if (PeeledFirst) {
    PreHeader->getTerminator()->setSuccessor(0, Header);

    for (auto &PHI : Header->phis())
      PHI.setIncomingValueForBlock(
          PreHeader, llvm::Constant::getIntegerValue(
                         PHI.getType(), llvm::APInt::getOneBitSet(PHI.getType()->getIntegerBitWidth(), 0)));

    NewBlocks.push_back(Header);
    VMap[WIHeader] = Header;
  } else {
    PreHeader->getTerminator()->setSuccessor(0, LoopBodyEntryBb);

    auto *BrCmpI = utils::getBrCmp(*Header);
    assert(BrCmpI && "WI Header must have cmp.");
    for (auto *BrOp : BrCmpI->operand_values()) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(BrOp)) {
        auto *LatchV = Phi->getIncomingValueForBlock(Latch);

        for (auto *U : Phi->users()) {
          if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
            if (UI->getParent() == Header)
              UI->replaceUsesOfWith(Phi, LatchV);
          }
        }

        // Move PHI from Header to for body
        Phi->moveBefore(&*LoopBodyEntryBb->begin());
        Phi->replaceIncomingBlockWith(Latch, Header);
        VMap[WIHeader] = LoopBodyEntryBb;
        break;
      }
    }

    // Header is now latch, so copy loop md over
    Header->getTerminator()->setMetadata("llvm.loop", Latch->getTerminator()->getMetadata("llvm.loop"));
    Latch->getTerminator()->setMetadata("llvm.loop", nullptr);
  }

  DT.reset();
  DT.recalculate(*F);

  /* Collect the basic blocks in the parallel region that dominate the
     exit. These are used in determining whether load instructions may
     be executed unconditionally in the parallel loop (see below). */
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> DominatesExitBb;
  for (auto *BB : Region) {
    if (DT.dominates(BB, ExitBb)) {
      DominatesExitBb.insert(BB);
    }
  }

  /* Fix the old edges jumping to the region to jump to the basic block
     that starts the created loop. Back edges should still point to the
     old basic block so we preserve the old loops. */
  BasicBlockVector Preds{llvm::pred_begin(EntryBb), llvm::pred_end(EntryBb)};

  for (auto *Bb : Preds) {
    /* Do not fix loop edges inside the region. The loop
       is replicated as a whole to the body of the wi-loop.*/
    if (DT.dominates(LoopBodyEntryBb, Bb))
      continue;
    Bb->getTerminator()->replaceUsesOfWith(LoopBodyEntryBb, PreHeader);
  }

  ExitBb->getTerminator()->replaceUsesOfWith(OldExit, Latch);

  Region.remap(VMap);
  llvm::remapInstructionsInBlocks(NewBlocks, VMap);

  return std::make_pair(PreHeader, utils::splitEdge(Header, OldExit, nullptr, &DT));
}

ParallelRegion *WorkItemLoopCreator::regionOfBlock(llvm::BasicBlock *BB) {
  for (auto *Region : *OriginalParallelRegions) {
    if (Region->HasBlock(BB))
      return Region;
  }
  return nullptr;
}

void WorkItemLoopCreator::releaseParallelRegions() {
  if (OriginalParallelRegions) {
    for (auto *P : *OriginalParallelRegions) {
      delete P;
    }

    delete OriginalParallelRegions;
    OriginalParallelRegions = nullptr;
  }
}

void WorkItemLoopCreator::getExitBlocks(llvm::Function &F, llvm::SmallVectorImpl<llvm::BasicBlock *> &ExitBlocks) {
  for (auto &BB : F) {
    auto *T = BB.getTerminator();
    if (T->getNumSuccessors() == 0) {
      // All exits must be barrier blocks.
      if (!utils::blockHasBarrier(&BB, SAA))
        utils::createBarrier(BB.getTerminator(), SAA);
      ExitBlocks.push_back(&BB);
    }
  }
}

/**
 * The main entry to the "parallel region formation", phase which search
 * for the regions between barriers that can be freely parallelized
 * across work-items in the work-group.
 */
ParallelRegion::ParallelRegionVector *WorkItemLoopCreator::getParallelRegions(llvm::Loop &L) {
  ParallelRegion::ParallelRegionVector *ParallelRegions = new ParallelRegion::ParallelRegionVector;

  auto *WILatch = L.getLoopLatch();
  llvm::SmallVector<llvm::BasicBlock *, 4> ExitBlocks{llvm::pred_begin(WILatch), llvm::pred_end(WILatch)};

  // We need to keep track of traversed barriers to detect back edges.
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> FoundBarriers;

  // First find all the ParallelRegions in the Function.
  while (!ExitBlocks.empty()) {

    // We start on an exit block and process the parallel regions upwards
    // (finding an execution trace).
    llvm::BasicBlock *Exit = ExitBlocks.back();
    ExitBlocks.pop_back();

    // already handled
    if (FoundBarriers.contains(Exit))
      continue;

    while (ParallelRegion *PR = createParallelRegionBefore(Exit)) {
      assert(PR != NULL && !PR->empty() && "Empty parallel region in kernel (contiguous barriers)!");

      FoundBarriers.insert(Exit);
      Exit = NULL;
      ParallelRegions->push_back(PR);
      llvm::BasicBlock *Entry = PR->entryBB();
      int FoundPredecessors = 0;
      llvm::BasicBlock *LoopBarrier = NULL;
      for (auto *Barrier : llvm::predecessors(Entry)) {
        if (!FoundBarriers.count(Barrier)) {
          /* If this is a loop header block we might have edges from two
             unprocessed barriers. The one inside the loop (coming from a
             computation block after a branch block) should be processed
             first. */
          std::string BbName = "";
          const bool IsInTheSameLoop =
              LI.getLoopFor(Barrier) && LI.getLoopFor(Entry) && LI.getLoopFor(Entry) == LI.getLoopFor(Barrier);

          if (IsInTheSameLoop) {
#ifdef DEBUG_PR_CREATION
            HIPSYCL_DEBUG_INFO << "### found a barrier inside the loop:" << Barrier->getName().str() << "\n";
#endif
            if (LoopBarrier != NULL) {
              // there can be multiple latches and each have their barrier,
              // save the previously found inner loop barrier
              ExitBlocks.push_back(LoopBarrier);
            }
            LoopBarrier = Barrier;
          } else {
#ifdef DEBUG_PR_CREATION
            HIPSYCL_DEBUG_INFO << "### found a barrier:" << Barrier->getName().str() << "\n";
#endif
            Exit = Barrier;
          }
          ++FoundPredecessors;
        }
      }

      if (LoopBarrier != NULL) {
        /* The secondary barrier to process in case it was a loop
           header. Push it for later processing. */
        if (Exit != NULL)
          ExitBlocks.push_back(Exit);
        /* always process the inner loop regions first */
        if (!FoundBarriers.count(LoopBarrier))
          Exit = LoopBarrier;
      }

#ifdef DEBUG_PR_CREATION
      HIPSYCL_DEBUG_INFO << "### created a ParallelRegion:\n";
      HIPSYCL_DEBUG_EXECUTE_INFO(PR->dumpNames();)
#endif

      if (FoundPredecessors == 0) {
        /* This path has been traversed and we encountered no more
           unprocessed regions. It means we have either traversed all
           paths from the exit or have transformed a loop and thus
           encountered only a barrier that was seen (and thus
           processed) before. */
        break;
      }
      assert((Exit != NULL) && "Parallel region without entry barrier!");
    }
  }

  return ParallelRegions;
}

ParallelRegion *WorkItemLoopCreator::createParallelRegionBefore(llvm::BasicBlock *B) {
  llvm::SmallVector<llvm::BasicBlock *, 4> PendingBlocks;
  llvm::SmallPtrSet<llvm::BasicBlock *, 8> BlocksInRegion;
  llvm::BasicBlock *RegionEntryBarrier = NULL;
  llvm::BasicBlock *Entry = NULL;
  llvm::BasicBlock *Exit = B->getSinglePredecessor();
  auto NotWILoopEntry = [WILoopEntry = this->WILoopEntry](llvm::BasicBlock *BB) {
    return BB != WILoopEntry->getSinglePredecessor();
  };
  addPredecessorsIf(PendingBlocks, B, NotWILoopEntry);

#ifdef DEBUG_PR_CREATION
  llvm::outs().SetUnbuffered();
  HIPSYCL_DEBUG_INFO << "createParallelRegionBefore " << B->getName() << "\n";
#endif

  while (!PendingBlocks.empty()) {
    llvm::BasicBlock *Current = PendingBlocks.back();
    PendingBlocks.pop_back();

#ifdef DEBUG_PR_CREATION
    HIPSYCL_DEBUG_INFO << "considering " << Current->getName() << "\n";
#endif

    // avoid infinite recursion of loops
    if (BlocksInRegion.count(Current) != 0) {
#ifdef DEBUG_PR_CREATION
      HIPSYCL_DEBUG_INFO << "already in the region!\n";
#endif
      continue;
    }

    // If we reach another barrier this must be the
    // parallel region entry.
    if (utils::hasOnlyBarrier(Current, SAA)) {
      if (RegionEntryBarrier == NULL)
        RegionEntryBarrier = Current;
#ifdef DEBUG_PR_CREATION
      HIPSYCL_DEBUG_INFO << "### it's a barrier!\n";
#endif
      continue;
    }

    assert(verifyNoBarriers(Current, SAA) &&
           "Barrier found in a non-barrier block! (forgot barrier canonicalization?)");

#ifdef DEBUG_PR_CREATION
    HIPSYCL_DEBUG_INFO << "added it to the region\n";
#endif
    // Non-barrier block, this must be on the region.
    BlocksInRegion.insert(Current);

    // Add predecessors to pending queue.
    addPredecessorsIf(PendingBlocks, Current, NotWILoopEntry);
  }

  if (BlocksInRegion.empty())
    return NULL;

  // Find the entry node.
  assert(RegionEntryBarrier != NULL);
  for (unsigned Suc = 0, Num = RegionEntryBarrier->getTerminator()->getNumSuccessors(); Suc < Num; ++Suc) {
    llvm::BasicBlock *EntryCandidate = RegionEntryBarrier->getTerminator()->getSuccessor(Suc);
    if (BlocksInRegion.count(EntryCandidate) == 0)
      continue;
    Entry = EntryCandidate;
    break;
  }
  assert(BlocksInRegion.count(Entry) != 0);

  // We got all the blocks in a region, create it.
  return ParallelRegion::Create(BlocksInRegion, Entry, Exit, SAA);
}

bool WorkItemLoopCreator::processFunction(llvm::Function &F) {
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  llvm::Module *M = F.getParent();

  releaseParallelRegions();

  WorkItemLoop = utils::getSingleWorkItemLoop(LI);
  assert(WorkItemLoop && "Kernel must have work item loop.");
  HIPSYCL_DEBUG_INFO << "WI loop header " << WorkItemLoop->getHeader()->getName() << "\n";
  WILoopEntry = utils::getWorkItemLoopBodyEntry(WorkItemLoop);
  WILoopExit = WorkItemLoop->getExitBlock();
  auto *WIIndVar = WorkItemLoop->getCanonicalInductionVariable();
  assert(WIIndVar);

  ContextArraySize = utils::getReqdStackElements(F);

  OriginalParallelRegions = getParallelRegions(*WorkItemLoop);
  fixMultiRegionAllocas(&F);

#ifdef DUMP_CFGS
  F.dump();
  dumpCFG(F, F.getName().str() + "_before_wiloops.dot", original_parallel_regions);
#endif

#if 0
  std::cerr << "### Original" << std::endl;
  F.viewCFGOnly();
#endif

  /* Count how many parallel regions share each entry node to
     detect diverging regions that need to be peeled. */
  std::map<llvm::BasicBlock *, int> EntryCounts;

  for (auto *Region : *OriginalParallelRegions) {

#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### Adding context save/restore for PR: ";
    HIPSYCL_DEBUG_EXECUTE_INFO(Region->dumpNames();)
#endif
    fixMultiRegionVariables(Region);
    EntryCounts[Region->entryBB()]++;
  }
  if (llvm::verifyFunction(F, &llvm::errs())) {
    HIPSYCL_DEBUG_ERROR << "function verification failed\n";
  }
#if 0
  std::cerr << "### After context code addition:" << std::endl;
  F.viewCFG();
#endif
  std::map<ParallelRegion *, bool> PeeledRegion;
  for (auto *Original : *OriginalParallelRegions) {

    llvm::ValueToValueMapTy ReferenceMap;

#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### handling region:\n";
    HIPSYCL_DEBUG_EXECUTE_INFO(Original->dumpNames();)
#endif

    /* In case of conditional barriers, the first iteration
       has to be peeled so we know which branch to execute
       with the work item loop. In case there are more than one
       parallel region sharing an entry BB, it's a diverging
       region.

       Post dominance of entry by exit does not work in case the
       region is inside a loop and the exit block is in the path
       towards the loop exit (and the function exit).
    */
    bool PeelFirst = EntryCounts[Original->entryBB()] > 1;

    PeeledRegion[Original] = PeelFirst;

    std::pair<llvm::BasicBlock *, llvm::BasicBlock *> L;
    // the original predecessor nodes of which successor
    // should be fixed if not peeling
    BasicBlockVector Preds;

    bool Unrolled = false;
    ParallelRegion *ToBeWrappedRegion = Original;
    if (PeelFirst) {
#ifdef DEBUG_WORK_ITEM_LOOPS
      HIPSYCL_DEBUG_INFO << "### conditional region, peeling the first iteration\n";
#endif
      ParallelRegion *Replica = Original->replicate(ReferenceMap, ".peeled_wi");
      Replica->chainAfter(Original);
      Replica->purge();

      L = std::make_pair(Replica->entryBB(), Replica->exitBB());
      ToBeWrappedRegion = Replica;
    } else {
      llvm::pred_iterator PI = llvm::pred_begin(Original->entryBB()), E = llvm::pred_end(Original->entryBB());

      for (; PI != E; ++PI) {
        llvm::BasicBlock *BB = *PI;
        if (DT.dominates(Original->entryBB(), BB) && (regionOfBlock(Original->entryBB()) == regionOfBlock(BB)))
          continue;
        Preds.push_back(BB);
      }
      // todo:

      //      unsigned UnrollCount;
      //      if (getenv("POCL_WILOOPS_MAX_UNROLL_COUNT") != NULL)
      //        UnrollCount = atoi(getenv("POCL_WILOOPS_MAX_UNROLL_COUNT"));
      //      else
      //        UnrollCount = 1;
      //      /* Find a two's exponent unroll count, if available. */
      //      while (UnrollCount >= 1) {
      //        if (WGLocalSizeX % unrollCount == 0 && unrollCount <= WGLocalSizeX) {
      //          break;
      //        }
      //        UnrollCount /= 2;
      //      }
      //
      //      if (UnrollCount > 1) {
      //        ParallelRegion *Prev = Original;
      //        llvm::BasicBlock *LastBb = AppendIncBlock(Original->exitBB(), LocalIdXGlobal);
      //        Original->AddBlockAfter(LastBb, Original->exitBB());
      //        Original->SetExitBB(LastBb);
      //
      //        if (AddWIMetadata)
      //          Original->AddIDMetadata(F.getContext(), 0);
      //
      //        for (unsigned c = 1; c < UnrollCount; ++c) {
      //          ParallelRegion *Unrolled = Original->replicate(ReferenceMap, ".unrolled_wi");
      //          Unrolled->chainAfter(Prev);
      //          Prev = Unrolled;
      //          LastBb = Unrolled->exitBB();
      //          if (AddWIMetadata)
      //            Unrolled->AddIDMetadata(F.getContext(), c);
      //        }
      //        Unrolled = true;
      //        L = std::make_pair(Original->entryBB(), LastBb);
      //      } else {
      L = std::make_pair(Original->entryBB(), Original->exitBB());
      //      }
    }

    L = createLoopAround(*ToBeWrappedRegion, L.first, L.second, PeelFirst);

    /* Loop edges coming from another region mean B-loops which means
       we have to fix the loop edge to jump to the beginning of the wi-loop
       structure, not its body. This has to be done only for non-peeled
       blocks as the semantics is correct in the other case (the jump is
       to the beginning of the peeled iteration). */
    if (!PeelFirst) {
      for (auto *BB : Preds) {
        BB->getTerminator()->replaceUsesOfWith(Original->entryBB(), L.first);
      }
    }
  }

  // for the peeled regions we need to add a prologue
  // that initializes the local ids and the first iteration
  // counter
  // replace idx with 0 for peeled
  llvm::ValueToValueMapTy VMap;
  llvm::PHINode *WIIndex = WorkItemLoop->getCanonicalInductionVariable();
  VMap.insert(std::make_pair(WIIndex, llvm::Constant::getNullValue(WIIndex->getType())));
  for (auto *Pr : *OriginalParallelRegions) {
    if (!PeeledRegion[Pr])
      continue;
    Pr->remap(VMap);
  }

  removeOriginalWILoop();

  return true;
}

void WorkItemLoopCreator::fixMultiRegionAllocas(llvm::Function *F) {
  InstructionVec InstructionsToFix;

  auto &LoopBlocks = WorkItemLoop->getBlocksSet();
  for (auto &I : F->getEntryBlock()) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (shouldNotBeContextSaved(&I) ||
          !std::all_of(Alloca->user_begin(), Alloca->user_end(), [&LoopBlocks](llvm::User *User) {
            const auto *Inst = llvm::dyn_cast<llvm::Instruction>(User);
            return Inst && LoopBlocks.contains(Inst->getParent());
          }))
        continue;
      InstructionsToFix.push_back(Alloca);
    }
  }
  for (auto *I : InstructionsToFix)
    addContextSaveRestore(I);
}

/*
 * Add context save/restore code to variables that are defined in
 * the given Region and are used outside the Region.
 *
 * Each such variable gets a slot in the stack frame. The variable
 * is restored from the stack whenever it's used.
 *
 */
void WorkItemLoopCreator::fixMultiRegionVariables(ParallelRegion *Region) {
  InstructionIndex InstructionsInRegion;
  InstructionVec InstructionsToFix;

  /* Construct an index of the Region's instructions so it's
     fast to figure out if the variable uses are all
     in the Region. */
  for (auto *BB : *Region) {
    for (auto &I : *BB) {
      InstructionsInRegion.insert(&I);
    }
  }

  /* Find all the instructions that define new values and
     check if they need to be context saved. */
  for (auto *Bb : *Region) {
    for (auto &I : *Bb) {
      if (shouldNotBeContextSaved(&I))
        continue;

      for (auto *User : I.users()) {
        auto *UI = llvm::dyn_cast<llvm::Instruction>(User);

        if (!UI)
          continue;
        // If the instruction is used outside this Region inside another
        // Region (not in a regionless BB like the B-loop construct BBs),
        // need to context save it.
        // Allocas (private arrays) should be privatized always. Otherwise
        // we end up reading the same array, but replicating the GEP to that.
        if (llvm::isa<llvm::AllocaInst>(I) ||
            (InstructionsInRegion.find(UI) == InstructionsInRegion.end() && regionOfBlock(UI->getParent()))) {
          InstructionsToFix.push_back(&I);
          break;
        }
      }
    }
  }

  /* Finally, fix the instructions. */
  for (auto *InstructionToFix : InstructionsToFix) {
    addContextSaveRestore(InstructionToFix);
  }
}

llvm::Instruction *WorkItemLoopCreator::addContextSave(llvm::Instruction *I, llvm::Instruction *Alloca) {

  if (llvm::isa<llvm::AllocaInst>(I)) {
    /* If the variable to be context saved is itself an alloca,
       we have created one big alloca that stores the data of all the
       work-items and return pointers to that array. Thus, we need
       no initialization code other than the context data alloca itself. */
    return nullptr;
  }

  /* Save the produced variable to the array. */
  auto Definition = I->getIterator();
  ++Definition;
  while (llvm::isa<llvm::PHINode>(Definition))
    ++Definition;

  llvm::IRBuilder<> Builder(&*Definition);
  std::vector<llvm::Value *> GepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *Region = regionOfBlock(I->getParent());
  assert("Adding context save outside any region produces illegal code." && Region != NULL);

  GepArgs.push_back(WorkItemLoop->getCanonicalInductionVariable());

  HIPSYCL_DEBUG_INFO << "loop header" << WorkItemLoop->getHeader()->getName() << "\n";
  HIPSYCL_DEBUG_INFO << "indvar: " << WorkItemLoop->getCanonicalInductionVariable()->getName() << "\n";
  HIPSYCL_DEBUG_INFO << Alloca->getName() << "\n";
  llvm::outs().flush();
  return Builder.CreateStore(I, Builder.CreateGEP(Alloca, GepArgs));
}

llvm::Instruction *WorkItemLoopCreator::addContextRestore(llvm::Value *Val, llvm::Instruction *Alloca,
                                                          bool PoclWrapperStructAdded, llvm::Instruction *Before,
                                                          bool IsAlloca) {
  assert(Val != NULL);
  assert(Alloca != NULL);
  llvm::IRBuilder<> Builder(Alloca);
  if (Before != NULL) {
    Builder.SetInsertPoint(Before);
  } else if (llvm::isa<llvm::Instruction>(Val)) {
    Builder.SetInsertPoint(llvm::dyn_cast<llvm::Instruction>(Val));
    Before = llvm::dyn_cast<llvm::Instruction>(Val);
  } else {
    assert(false && "Unknown context restore location!");
  }

  std::vector<llvm::Value *> GepArgs;

  /* Reuse the id loads earlier in the region, if possible, to
     avoid messy output with lots of redundant loads. */
  ParallelRegion *Region = regionOfBlock(Before->getParent());
  assert("Adding context save outside any region produces illegal code." && Region != NULL);

  assert(WorkItemLoop->getCanonicalInductionVariable());
  GepArgs.push_back(WorkItemLoop->getCanonicalInductionVariable());

  if (PoclWrapperStructAdded)
    GepArgs.push_back(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Alloca->getContext()), 0));
  assert(Alloca && GepArgs.size() && "beeeeeeep");

  llvm::Instruction *GEP = llvm::dyn_cast<llvm::Instruction>(Builder.CreateGEP(Alloca, GepArgs));
  if (IsAlloca) {
    /* In case the context saved instruction was an alloca, we created a
       context array with pointed-to elements, and now want to return a
       pointer to the elements to emulate the original alloca. */
    return GEP;
  }
  return Builder.CreateLoad(GEP);
}

/**
 * Returns the context array (alloca) for the given Value, creates it if not
 * found.
 */
llvm::Instruction *WorkItemLoopCreator::getContextArray(llvm::Instruction *I, bool &PoclWrapperStructAdded) {
  PoclWrapperStructAdded = false;
  /*
   * Unnamed temp instructions need a generated name for the
   * context array. Create one using a running integer.
   */
  std::ostringstream Var;
  Var << ".";

  if (std::string(I->getName().str()) != "") {
    Var << I->getName().str();
  } else if (TempInstructionIds.find(I) != TempInstructionIds.end()) {
    Var << TempInstructionIds[I];
  } else {
    TempInstructionIds[I] = TempInstructionIndex++;
    Var << TempInstructionIds[I];
  }

  Var << ".pocl_context";
  std::string VarName = Var.str();

  if (ContextArrays.find(VarName) != ContextArrays.end())
    return ContextArrays[VarName];

  llvm::BasicBlock &BB = I->getParent()->getParent()->getEntryBlock();
  llvm::IRBuilder<> Builder(&*(BB.getFirstInsertionPt()));
  llvm::Function *FF = I->getParent()->getParent();
  llvm::Module *M = I->getParent()->getParent()->getParent();
  llvm::LLVMContext &C = M->getContext();
  const llvm::DataLayout &Layout = M->getDataLayout();
  llvm::DICompileUnit *CU = nullptr;
  std::unique_ptr<llvm::DIBuilder> DB;

  // find the debug metadata corresponding to this variable
  llvm::Value *DebugVal = nullptr;
  llvm::IntrinsicInst *DebugCall = nullptr;

#ifdef DEBUG_WORK_ITEM_LOOPS
  if (DebugVal && DebugCall) {
    HIPSYCL_DEBUG_INFO << "### DI INTRIN: \n";
    HIPSYCL_DEBUG_EXECUTE_INFO(DebugCall->print(llvm::outs()); llvm::outs() << "\n";)
    HIPSYCL_DEBUG_INFO << "### DI VALUE:  \n";
    HIPSYCL_DEBUG_EXECUTE_INFO(DebugVal->print(llvm::outs()); llvm::outs() << "\n";)
  }
#endif

  llvm::Type *ElementType;
  if (llvm::isa<llvm::AllocaInst>(I)) {
    /* If the variable to be context saved was itself an alloca,
       create one big alloca that stores the data of all the
       work-items and directly return pointers to that array.
       This enables moving all the allocas to the entry node without
       breaking the parallel loop.
       Otherwise we would rely on a dynamic alloca to allocate
       unique stack space to all the work-items when its wiloop
       iteration is executed. */
    ElementType = llvm::dyn_cast<llvm::AllocaInst>(I)->getType()->getElementType();
  } else {
    ElementType = I->getType();
  }

  /* 3D context array. In case the elementType itself is an array or struct,
   * we must take into account it could be alloca-ed with alignment and loads
   * or stores might use vectorized instructions expecting proper alignment.
   * Because of that, we cannot simply allocate x*y*z*(size), we must
   * enlarge the type to fit the alignment. */
  llvm::Type *AllocType = ElementType;
  llvm::AllocaInst *InstCast = llvm::dyn_cast<llvm::AllocaInst>(I);
  if (InstCast) {
    unsigned Alignment = InstCast->getAlignment();

    uint64_t StoreSize = Layout.getTypeStoreSize(InstCast->getAllocatedType());

    if ((Alignment > 1) && (StoreSize & (Alignment - 1))) {
      uint64_t AlignedSize = (StoreSize & (~(Alignment - 1))) + Alignment;
#ifdef DEBUG_WORK_ITEM_LOOPS
      HIPSYCL_DEBUG_INFO << "### unaligned type found: aligning " << StoreSize << " to " << AlignedSize << "\n";
#endif
      assert(AlignedSize > StoreSize);
      uint64_t RequiredExtraBytes = AlignedSize - StoreSize;

      if (llvm::isa<llvm::ArrayType>(ElementType)) {

        llvm::ArrayType *StructPadding =
            llvm::ArrayType::get(llvm::Type::getInt8Ty(M->getContext()), RequiredExtraBytes);

        std::vector<llvm::Type *> PaddedStructElements;
        PaddedStructElements.push_back(ElementType);
        PaddedStructElements.push_back(StructPadding);
        const llvm::ArrayRef<llvm::Type *> NewStructElements(PaddedStructElements);
        AllocType = llvm::StructType::get(M->getContext(), NewStructElements, true);
        PoclWrapperStructAdded = true;
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);

      } else if (llvm::isa<llvm::StructType>(ElementType)) {
        llvm::StructType *OldStruct = llvm::dyn_cast<llvm::StructType>(ElementType);

        llvm::ArrayType *StructPadding =
            llvm::ArrayType::get(llvm::Type::getInt8Ty(M->getContext()), RequiredExtraBytes);
        std::vector<llvm::Type *> PaddedStructElements;
        for (unsigned J = 0; J < OldStruct->getNumElements(); J++)
          PaddedStructElements.push_back(OldStruct->getElementType(J));
        PaddedStructElements.push_back(StructPadding);
        const llvm::ArrayRef<llvm::Type *> NewStructElements(PaddedStructElements);
        AllocType = llvm::StructType::get(OldStruct->getContext(), NewStructElements, OldStruct->isPacked());
        uint64_t NewStoreSize = Layout.getTypeStoreSize(AllocType);
        assert(NewStoreSize == AlignedSize);
      }
    }
  }

  llvm::AllocaInst *Alloca = nullptr;

  // todo: can we make the stack usage smaller by using runtime values? does that impact vectorization?
  //    llvm::Value *LocalXTimesY = Builder.CreateBinOp(llvm::Instruction::Mul, LocalSizeLoad[0], LocalSizeLoad[1],
  //    "tmp"); llvm::Value *NumberOfWorkItems =
  //        Builder.CreateBinOp(llvm::Instruction::Mul, LocalXTimesY, LocalSizeLoad[2], "num_wi");

  Alloca = Builder.CreateAlloca(AllocType, Builder.getInt32(ContextArraySize), VarName);

  /* Align the context arrays to stack to enable wide vectors
     accesses to them. Also, LLVM 3.3 seems to produce illegal
     code at least with Core i5 when aligned only at the element
     size. */
  Alloca->setAlignment(llvm::Align(CONTEXT_ARRAY_ALIGN));

  ContextArrays[VarName] = Alloca;
  return Alloca;
}

/**
 * Adds context save/restore code for the value produced by the
 * given I.
 *
 * TODO: add only one restore per variable per region.
 * TODO: add only one load of the id variables per region.
 * Could be done by having a context restore BB in the beginning of the
 * region and a context save BB at the end.
 * TODO: ignore work group variables completely (the iteration variables)
 * The LLVM should optimize these away but it would improve
 * the readability of the output during debugging.
 * TODO: rematerialize some values such as extended values of global
 * variables (especially global id which is computed from local id) or kernel
 * argument values instead of allocating stack space for them
 */
void WorkItemLoopCreator::addContextSaveRestore(llvm::Instruction *I) {
#ifdef DEBUG_WORK_ITEM_LOOPS
  HIPSYCL_DEBUG_INFO << "### adding context/save restore for: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(I->print(llvm::outs()); llvm::outs() << "\n";)
#endif
  /* Allocate the context data array for the variable. */
  bool PoclWrapperStructAdded = false;
  llvm::Instruction *Alloca = getContextArray(I, PoclWrapperStructAdded);
  llvm::Instruction *TheStore = addContextSave(I, Alloca);

  InstructionVec Uses;
  /* Restore the produced variable before each use to ensure the correct context
     copy is used.

     We could add the restore only to other regions outside the
     variable defining region and use the original variable in the defining
     region due to the SSA virtual registers being unique. However,
     alloca variables can be redefined also in the same region, thus we
     need to ensure the correct alloca context position is written, not
     the original unreplicated one. These variables can be generated by
     volatile variables, private arrays, and due to the PHIs to allocas
     pass.
  */

  /* Find out the uses to fix first as fixing them invalidates
     the iterator. */
  for (auto *User : I->users()) {
    auto *UI = llvm::cast<llvm::Instruction>(User);
    if (!UI)
      continue;
    if (UI == TheStore)
      continue;
    Uses.push_back(UI);
  }

  for (auto *User : Uses) {
    auto *ContextRestoreLocation = User;
    /* If the User is in a block that doesn't belong to a region,
       the variable itself must be a "work group variable", that is,
       not dependent on the work item. Most likely an iteration
       variable of a for loop with a barrier. */
    if (!regionOfBlock(User->getParent()))
      continue;

    llvm::PHINode *Phi = llvm::dyn_cast<llvm::PHINode>(User);
    if (Phi) {
      /* In case of PHI nodes, we cannot just insert the context
         restore code before it in the same basic block because it is
         assumed there are no non-phi Instructions before PHIs which
         the context restore code constitutes to. Add the context
         restore to the incomingBB instead.

         There can be values in the PHINode that are incoming
         from another region even though the decision BB is within the region.
         For those values we need to add the context restore code in the
         incoming BB (which is known to be inside the region due to the
         assumption of not having to touch PHI nodes in PRentry BBs).
      */

      /* PHINodes at region entries are broken down earlier. */
      assert("Cannot add context restore for a PHI node at the region entry!" &&
             regionOfBlock(Phi->getParent())->entryBB() != Phi->getParent());
#ifdef DEBUG_WORK_ITEM_LOOPS
      HIPSYCL_DEBUG_INFO << "### adding context restore code before PHI\n";
      HIPSYCL_DEBUG_EXECUTE_INFO(User->print(llvm::outs()); llvm::outs() << "\n";)
      HIPSYCL_DEBUG_INFO << "### in BB:\n";
      HIPSYCL_DEBUG_EXECUTE_INFO(User->getParent()->print(llvm::outs()); llvm::outs() << "\n";)
#endif
      llvm::BasicBlock *IncomingBb = NULL;
      for (unsigned Incoming = 0; Incoming < Phi->getNumIncomingValues(); ++Incoming) {
        llvm::Value *Val = Phi->getIncomingValue(Incoming);
        llvm::BasicBlock *Bb = Phi->getIncomingBlock(Incoming);
        if (Val == I)
          IncomingBb = Bb;
      }
      assert(IncomingBb != NULL);
      ContextRestoreLocation = IncomingBb->getTerminator();
    }
    llvm::Value *LoadedValue =
        addContextRestore(User, Alloca, PoclWrapperStructAdded, ContextRestoreLocation, llvm::isa<llvm::AllocaInst>(I));
    User->replaceUsesOfWith(I, LoadedValue);

#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### done, the User was converted to:\n";
    HIPSYCL_DEBUG_EXECUTE_INFO(User->print(llvm::outs()); llvm::outs() << "\n";)
#endif
  }
}

bool WorkItemLoopCreator::shouldNotBeContextSaved(llvm::Instruction *Instr) {
  /*
    _local_id loads should not be replicated as it leads to
    problems in conditional branch case where the header node
    of the region is shared across the branches and thus the
    header node's ID loads might get context saved which leads
    to egg-chicken problems.
  */
  if (llvm::isa<llvm::BranchInst>(Instr))
    return true;

  /* In case of uniform variables (same for all work-items),
     there is no point to create a context array slot for them,
     but just use the original value everywhere.

     Allocas are problematic: they include the de-phi induction
     variables of the b-loops. In those case each work item
     has a separate loop iteration variable in the LLVM IR but
     which is really a parallel region loop invariant. But
     because we cannot separate such loop invariant variables
     at this point sensibly, let's just replicate the iteration
     variable to each work item and hope the latter optimizations
     reduce them back to a single induction variable outside the
     parallel loop.
  */
  if (!VUA.shouldBePrivatized(Instr->getParent()->getParent(), Instr)) {
#ifdef DEBUG_WORK_ITEM_LOOPS
    HIPSYCL_DEBUG_INFO << "### based on VUA, not context saving:";
    HIPSYCL_DEBUG_EXECUTE_INFO(Instr->print(llvm::outs()); llvm::outs() << "\n";)
#endif
    return true;
  }

  return false;
}

bool dominatesUse(llvm::DominatorTree &DT, llvm::Instruction &I, unsigned Idx) {
  llvm::Instruction *Op = llvm::cast<llvm::Instruction>(I.getOperand(Idx));
  llvm::BasicBlock *OpBlock = Op->getParent();
  llvm::PHINode *PN = llvm::dyn_cast<llvm::PHINode>(&I);

  // DT can handle non phi instructions for us.
  if (!PN) {
    // Definition must dominate use unless use is unreachable!
    return Op->getParent() == I.getParent() || DT.dominates(Op, &I);
  }

  // PHI nodes are more difficult than other nodes because they actually
  // "use" the value in the predecessor basic blocks they correspond to.
  unsigned J = llvm::PHINode::getIncomingValueNumForOperand(Idx);
  llvm::BasicBlock *PredBB = PN->getIncomingBlock(J);
  return (PredBB && DT.dominates(OpBlock, PredBB));
}

/* Fixes the undominated variable uses.

   These appear when a conditional barrier kernel is replicated to
   form a copy of the *same basic block* in the alternative
   "barrier path".

   E.g., from

   A -> [exit], A -> B -> [exit]

   a replicated CFG as follows, is created:

   A1 -> (T) A2 -> [exit1],  A1 -> (F) A2' -> B1, B2 -> [exit2]

   The regions are correct because of the barrier semantics
   of "all or none". In case any barrier enters the [exit1]
   from A1, all must (because there's a barrier in the else
   branch).

   Here at A2 and A2' one creates the same variables.
   However, B2 does not know which copy
   to refer to, the ones created in A2 or ones in A2' (correct).
   The mapping data contains only one possibility, the
   one that was placed there last. Thus, the instructions in B2
   might end up referring to the variables defined in A2
   which do not nominate them.

   The variable references are fixed by exploiting the knowledge
   of the naming convention of the cloned variables.

   One potential alternative way would be to collect the refmaps per BB,
   not globally. Then as a final phase traverse through the
   basic blocks starting from the beginning and propagating the
   reference data downwards, the data from the new BB overwriting
   the old one. This should ensure the reachability without
   the costly dominance analysis.
*/
bool WorkItemLoopCreator::fixUndominatedVariableUses(llvm::Function &F) {
  bool Changed = false;

  DT.reset();
  DT.recalculate(F);

  for (llvm::Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    llvm::BasicBlock *Bb = &*I;
    for (auto &Ins : *Bb) {
      for (unsigned Opr = 0; Opr < Ins.getNumOperands(); ++Opr) {
        if (!llvm::isa<llvm::Instruction>(Ins.getOperand(Opr)))
          continue;
        llvm::Instruction *Operand = llvm::cast<llvm::Instruction>(Ins.getOperand(Opr));
        if (dominatesUse(DT, Ins, Opr))
          continue;
#ifdef DEBUG_REFERENCE_FIXING
        HIPSYCL_DEBUG_INFO << "### dominance error!\n";
        HIPSYCL_DEBUG_EXECUTE_INFO(Operand->print(llvm::outs()); llvm::outs() << "\n";)
        HIPSYCL_DEBUG_INFO << "### does not dominate:\n";
        HIPSYCL_DEBUG_EXECUTE_INFO(Ins.print(llvm::outs()); llvm::outs() << "\n";)
#endif
        llvm::StringRef BaseName;
        std::pair<llvm::StringRef, llvm::StringRef> Pieces = Operand->getName().rsplit('.');
        if (Pieces.second.startswith("pocl_"))
          BaseName = Pieces.first;
        else
          BaseName = Operand->getName();

        llvm::Value *Alternative = NULL;

        unsigned int CopyI = 0;
        do {
          std::ostringstream AlternativeName;
          AlternativeName << BaseName.str();
          if (CopyI > 0)
            AlternativeName << ".pocl_" << CopyI;

          Alternative = F.getValueSymbolTable()->lookup(AlternativeName.str());

          if (Alternative != NULL) {
            Ins.setOperand(Opr, Alternative);
            if (dominatesUse(DT, Ins, Opr))
              break;
          }

          if (CopyI > 10000 && Alternative == NULL)
            break; /* ran out of possibilities */
          ++CopyI;
        } while (true);

        if (Alternative != NULL) {
#ifdef DEBUG_REFERENCE_FIXING
          HIPSYCL_DEBUG_INFO << "### found the alternative:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alternative->print(llvm::outs()); llvm::outs() << "\n";)
#endif
          Changed |= true;
        } else {
#ifdef DEBUG_REFERENCE_FIXING
          HIPSYCL_DEBUG_INFO << "### didn't find an alternative for\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Operand->print(llvm::outs()); llvm::outs() << "\n";)
          HIPSYCL_DEBUG_INFO << "### BB:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Operand->getParent()->print(llvm::outs()); llvm::outs() << "\n";)
          HIPSYCL_DEBUG_INFO << "### the user BB:\n";
          HIPSYCL_DEBUG_EXECUTE_INFO(Ins.getParent()->print(llvm::outs()); llvm::outs() << "\n";)
#endif
          llvm::outs().flush();
          std::cerr << "Could not find a dominating alternative variable." << std::endl;
          //          dumpCFG(F, "broken.dot");
          abort();
        }
      }
    }
  }
  return Changed;
}

WorkItemLoopCreator::WorkItemLoopCreator(llvm::Function &F, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                                         llvm::LoopInfo &LI, VariableUniformityInfo &VUA, SplitterAnnotationInfo &SAA)
    : OriginalParallelRegions(nullptr), DT(DT), PDT(PDT), LI(LI), VUA(VUA), SAA(SAA), TempInstructionIndex(0) {
  llvm::Module *M = F.getParent();
  llvm::DataLayout DL(M);
  SizeT = DL.getLargestLegalIntType(M->getContext());
}

void replacePredecessorsSuccessor(llvm::BasicBlock *Old, llvm::BasicBlock *New) {
  llvm::SmallVector<llvm::BasicBlock *, 4> Preds{llvm::pred_begin(Old), llvm::pred_end(Old)};

  for (auto *Bb : Preds) {
    Bb->getTerminator()->replaceUsesOfWith(Old, New);
  }
}

void WorkItemLoopCreator::removeOriginalWILoop() {
  auto *PreHeader = WorkItemLoop->getLoopPreheader();
  auto *Header = WorkItemLoop->getHeader();
  auto *Latch = WorkItemLoop->getLoopLatch();

  auto *IndVar = WorkItemLoop->getCanonicalInductionVariable();
  IndVar->replaceAllUsesWith(llvm::Constant::getNullValue(IndVar->getType()));

  replacePredecessorsSuccessor(PreHeader, WILoopEntry);
  replacePredecessorsSuccessor(Latch, WILoopExit);

  Header->replaceAllUsesWith(WILoopExit);
  Header->eraseFromParent();
  Latch->replaceAllUsesWith(WILoopExit);
  Latch->eraseFromParent();
  PreHeader->replaceAllUsesWith(WILoopExit);
  PreHeader->eraseFromParent();
}
} // namespace

namespace hipsycl::compiler {
char WorkItemLoopCreationPassLegacy::ID = 0;

void WorkItemLoopCreationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::PostDominatorTreeWrapperPass>();

  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequired<llvm::DominatorTreeWrapperPass>();

  AU.addRequired<VariableUniformityAnalysisLegacy>();
  AU.addPreserved<VariableUniformityAnalysisLegacy>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool WorkItemLoopCreationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &VUA = getAnalysis<VariableUniformityAnalysisLegacy>().getResult();

  WorkItemLoopCreator WILC{F, DT, PDT, LI, VUA, SAA};
  bool Changed = WILC.processFunction(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  Changed |= WILC.fixUndominatedVariableUses(F);

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  return Changed;
}

llvm::PreservedAnalyses WorkItemLoopCreationPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());

  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA)) {
    return llvm::PreservedAnalyses::all();
  }

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &VUA = AM.getResult<VariableUniformityAnalysis>(F);

  WorkItemLoopCreator WILC{F, DT, PDT, LI, VUA, *SAA};

  bool Changed = WILC.processFunction(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  Changed |= WILC.fixUndominatedVariableUses(F);
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  if (!Changed)
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
