// Class definition for parallel regions, a group of BasicBlocks that
// each kernel should run in parallel.
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
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

#include "hipSYCL/compiler/ParallelRegion.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

//#define DEBUG_REMAP
//#define DEBUG_REPLICATE
//#define DEBUG_PURGE

#include <algorithm>
#include <iostream>
#include <map>
#include <set>

namespace hipsycl::compiler {

int ParallelRegion::idGen = 0;

ParallelRegion::ParallelRegion(const SplitterAnnotationInfo &SAA, int ForcedRegionId)
    : LocalIDXLoadInstr(NULL), LocalIDYLoadInstr(NULL), LocalIDZLoadInstr(NULL), exitIndex_(0), entryIndex_(0),
      SAA(SAA), pRegionId(ForcedRegionId) {
  if (ForcedRegionId == -1)
    pRegionId = idGen++;
}

/**
 * Ensure all variables are named so they will be replicated and renamed
 * correctly.
 */
void ParallelRegion::GenerateTempNames(llvm::BasicBlock *Bb) {
  for (llvm::Instruction &I : *Bb) {
    if (I.hasName() || !I.isUsedOutsideOfBlock(Bb))
      continue;
    int TempCounter = 0;
    std::string TempName = "";
    do {
      std::ostringstream Name;
      Name << ".pocl_temp." << TempCounter;
      ++TempCounter;
      TempName = Name.str();
    } while (Bb->getParent()->getValueSymbolTable()->lookup(TempName));
    I.setName(TempName);
  }
}

ParallelRegion *ParallelRegion::replicate(llvm::ValueToValueMapTy &VMap, const llvm::Twine &Suffix = "") {
  ParallelRegion *NewRegion = new ParallelRegion(SAA, pRegionId);

  /* Because ParallelRegions are all replicated before they
     are attached to the function, it can happen that
     the same BB is replicated multiple times and it gets
     the same name (only the BB name will be autorenamed
     by LLVM). This causes the variable references to become
     broken. This hack ensures the BB suffixes are unique
     before cloning so each path gets their own value
     names. Split points can be such paths.*/
  static std::map<std::string, int> CloneCounts;

  for (auto *Block : *this) {
    GenerateTempNames(Block);
    std::ostringstream Suf;
    Suf << Suffix.str();
    std::string BlockName = Block->getName().str() + "." + Suffix.str();
    if (CloneCounts[BlockName] > 0) {
      Suf << ".pocl_" << CloneCounts[BlockName];
    }
    llvm::BasicBlock *NewBlock = CloneBasicBlock(Block, VMap, Suf.str());
    CloneCounts[BlockName]++;
    // Insert the block itself into the VMap.
    VMap[Block] = NewBlock;
    NewRegion->push_back(NewBlock);

#ifdef DEBUG_REPLICATE
    std::cerr << "### clonee block:" << std::endl;
    Block->dump();
    std::cerr << endl << "### cloned block: " << std::endl;
    new_block->dump();
#endif
  }

  NewRegion->exitIndex_ = exitIndex_;
  NewRegion->entryIndex_ = entryIndex_;
  /* Remap here to get local variables fixed before they
     are (possibly) overwritten by another clone of the
     same BB. */
  NewRegion->remap(VMap);

#ifdef DEBUG_REPLICATE
  Verify();
#endif
//  LocalizeIDLoads();

  return NewRegion;
}

void ParallelRegion::remap(llvm::ValueToValueMapTy &Map) {
  for (auto &BB : *this) {

#ifdef DEBUG_REMAP
    std::cerr << "### block before remap:" << std::endl;
    (*BB)->dump();
#endif

    for (auto &I : *BB)
      RemapInstruction(&I, Map, llvm::RF_IgnoreMissingLocals | llvm::RF_NoModuleLevelChanges);

#ifdef DEBUG_REMAP
    std::cerr << endl << "### block after remap: " << std::endl;
    (*BB)->dump();
#endif
  }
}

void ParallelRegion::chainAfter(ParallelRegion *Region) {
  /* If we are replicating a conditional barrier Region, the last block can be
     an unreachable block to mark the impossible path. Skip it and choose the
     correct branch instead.

     TODO: why have the unreachable block there the first place? Could we just
     not add it and fix the branch? */
  llvm::BasicBlock *Tail = Region->exitBB();
  auto *T = Tail->getTerminator();
  if (llvm::isa<llvm::UnreachableInst>(T)) {
    Tail = Region->at(Region->size() - 2);
    T = Tail->getTerminator();
  }
#ifdef LLVM_BUILD_MODE_DEBUG
  if (T->getNumSuccessors() != 1) {
    std::cout << "!!! trying to chain Region" << std::endl;
    this->dumpNames();
    std::cout << "!!! after Region" << std::endl;
    Region->dumpNames();
    T->getParent()->dump();

    assert(T->getNumSuccessors() == 1);
  }
#endif

  llvm::BasicBlock *Successor = T->getSuccessor(0);
  auto &BBList = Successor->getParent()->getBasicBlockList();

  for (auto &I : *this)
    BBList.insertAfter(Tail->getIterator(), I);

  T->setSuccessor(0, entryBB());

  T = exitBB()->getTerminator();
  assert(T->getNumSuccessors() == 1);
  T->setSuccessor(0, Successor);
}

/**
 * Removes known dead side exits from parallel regions.
 *
 * These occur with conditional barriers. The head of the path
 * leading to the conditional barrier is shared by two PRs. The
 * first work-item defines which path is taken (by definition the
 * barrier is taken by all or none of the work-items). The blocks
 * in the branches are in different regions which can contain branches
 * to blocks that are in known non-taken path. This method replaces
 * the targets of such branches with undefined BBs so they will be cleaned
 * up by the optimizer.
 */
void ParallelRegion::purge() {
  llvm::SmallVector<llvm::BasicBlock *, 4> NewBlocks;

  // Go through all the BBs in the region and check their branch
  // targets, looking for destinations that are outside the region.
  // Only the last block in the PR can now contain such branches.
  for (auto *BB : *this) {

    // Exit block has a successor out of the region.
    if (BB == exitBB())
      continue;

#ifdef DEBUG_PURGE
    std::cerr << "### block before purge:" << std::endl;
    (*i)->dump();
#endif
    auto *T = BB->getTerminator();
    for (unsigned SuccIdx = 0, NumSucc = T->getNumSuccessors(); SuccIdx != NumSucc; ++SuccIdx) {
      llvm::BasicBlock *Successor = T->getSuccessor(SuccIdx);
      if (count(begin(), end(), Successor) == 0) {
        // This successor is not on the parallel region, purge.
#ifdef DEBUG_PURGE
        std::cerr << "purging a branch to a block " << successor->getName().str() << " outside the region" << std::endl;
#endif

        llvm::BasicBlock *Unreachable =
            llvm::BasicBlock::Create(BB->getContext(), BB->getName() + ".unreachable", BB->getParent(), back());
        new llvm::UnreachableInst(Unreachable->getContext(), Unreachable);
        T->setSuccessor(SuccIdx, Unreachable);
        NewBlocks.push_back(Unreachable);
      }
    }
#ifdef DEBUG_PURGE
    std::cerr << std::endl << "### block after purge:" << std::endl;
    (*i)->dump();
#endif
  }

  // Add the new "unreachable" blocks to the
  // region. We cannot do in the loop as it
  // corrupts iterators.
  insert(end(), NewBlocks.begin(), NewBlocks.end());
}

void ParallelRegion::dump() {
#ifdef LLVM_BUILD_MODE_DEBUG
  for (iterator i = begin(), e = end(); i != e; ++i)
    (*i)->dump();
#endif
}

void ParallelRegion::dumpNames() {
  for (auto &BB : BBs_) {
    llvm::outs() << BB->getName().str();
    if (entryBB() == BB)
      llvm::outs() << "(EN)";
    if (exitBB() == BB)
      llvm::outs() << "(EX)";
    llvm::outs() << " ";
  }
  llvm::outs() << "\n";
}

ParallelRegion *ParallelRegion::Create(const llvm::SmallPtrSet<llvm::BasicBlock *, 8> &BBs, llvm::BasicBlock *Entry,
                                       llvm::BasicBlock *Exit, const SplitterAnnotationInfo &SAA) {
  ParallelRegion *NewRegion = new ParallelRegion(SAA);

  assert(Entry != NULL);
  assert(Exit != NULL);

  // This is done in two steps so order of the vector
  // is the same as original function order.
  llvm::Function *F = Entry->getParent();
  for (auto &FBB : *F) {
    for (auto *BB : BBs) {
      if (BB == &FBB) {
        NewRegion->push_back(&FBB);
        if (Entry == BB)
          NewRegion->setEntryBBIndex(NewRegion->size() - 1);
        else if (Exit == BB)
          NewRegion->setExitBBIndex(NewRegion->size() - 1);
        break;
      }
    }
  }

  assert(NewRegion->Verify());

  return NewRegion;
}

bool ParallelRegion::Verify() {
  // Parallel region conditions:
  // 1) Single entry, in entry block.
  // 2) Single outgoing edge from exit block
  //    (other outgoing edges allowed, will be purged in replicas).
  // 3) No barriers inside the region.

  int EntryEdges = 0;

  for (auto *BB : *this) {
    for (auto *PredBB : llvm::predecessors(BB)) {
      if (count(begin(), end(), PredBB) == 0) {
        if (BB != entryBB()) {
          dumpNames();
          std::cerr << "suspicious block: " << BB->getName().str() << std::endl;
          std::cerr << "the entry is: " << entryBB()->getName().str() << std::endl;

          ParallelRegion::ParallelRegionVector PRs;
          PRs.push_back(this);
          std::set<llvm::BasicBlock *> Highlights;
          Highlights.insert(entryBB());
          Highlights.insert(BB);

          assert(false && "Incoming edges to non-entry block!");
          return false;
        }
        if (!hipsycl::compiler::utils::blockHasBarrier(PredBB, SAA)) {
          BB->getParent()->viewCFG();
          assert(false && "Entry has edges from non-barrier blocks!");
          return false;
        }
        ++EntryEdges;
      }
    }

    // if (entry_edges != 1) {
    //   assert(0 && "Parallel regions must be single entry!");
    //   return false;
    // }
    if (exitBB()->getTerminator()->getNumSuccessors() != 1) {
      ParallelRegion::ParallelRegionVector Regions;
      Regions.push_back(this);

#ifdef LLVM_BUILD_MODE_DEBUG
      std::set<llvm::BasicBlock *> highlights;
      highlights.insert((*i));
      highlights.insert(exitBB());
      exitBB()->dump();
      dumpNames();
      dumpCFG(*(*i)->getParent(), "broken.dot", &regions, &highlights);
#endif

      assert(false && "Multiple outgoing edges from exit block!");
      return false;
    }

    if (hipsycl::compiler::utils::blockHasBarrier(BB, SAA)) {
      assert(false && "Barrier found inside parallel region!");
      return false;
    }
  }

  return true;
}

#define PARALLEL_MD_NAME "llvm.access.group"

/**
 * Adds metadata to all the memory instructions to denote
 * they originate from a parallel loop.
 *
 * Due to nested parallel loops, there can be multiple loop
 * references.
 *
 * Format (LLVM 8+):
 *
 *     !llvm.access.group !0
 *
 *     !0 distinct !{}
 *
 * In a 2-nested loop:
 *
 *     !llvm.access.group !0
 *
 *     !0 { !1, !2 }
 *     !1 distinct !{}
 *     !2 distinct !{}
 *
 * Parallel loop metadata on memory reads also implies that
 * if-conversion (i.e., speculative execution within a loop iteration)
 * is safe. Given an instruction reading from memory,
 * IsLoadUnconditionallySafe should return whether it is safe under
 * (unconditional, unpredicated) speculative execution.
 * See https://bugs.llvm.org/show_bug.cgi?id=46666
 */
void ParallelRegion::AddParallelLoopMetadata(llvm::MDNode *Identifier,
                                             std::function<bool(llvm::Instruction *)> IsLoadUnconditionallySafe) {
  for (auto *BB : *this) {
    for (auto &I : *BB) {
      if (!I.mayReadOrWriteMemory()) {
        continue;
      }
#if LLVM_VERSION_MAJOR < 13 && !(LLVM_VERSION_MAJOR == 12 && LLVM_VERSION_MINOR >= 0 && LLVM_VERSION_PATCH >= 1)
      if (I.mayReadFromMemory() && !IsLoadUnconditionallySafe(&I)) {
        continue;
      }
#endif

      llvm::MDNode *NewMD = llvm::MDNode::get(BB->getContext(), Identifier);
      llvm::MDNode *OldMD = I.getMetadata(PARALLEL_MD_NAME);
      if (OldMD != nullptr) {
        NewMD = llvm::MDNode::concatenate(OldMD, NewMD);
      }
      I.setMetadata(PARALLEL_MD_NAME, NewMD);
    }
  }
}

void ParallelRegion::AddIDMetadata(llvm::LLVMContext &Context, std::size_t X, std::size_t Y, std::size_t Z) {
  int Counter = 1;
  llvm::Metadata *V1[] = {
      llvm::MDString::get(Context, "WI_region"),
      llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), pRegionId))};
  llvm::MDNode *MdRegion = llvm::MDNode::get(Context, V1);
  llvm::Metadata *V2[] = {llvm::MDString::get(Context, "WI_xyz"),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), X)),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), Y)),
                          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), Z))};
  llvm::MDNode *MdXyz = llvm::MDNode::get(Context, V2);
  llvm::Metadata *V[] = {llvm::MDString::get(Context, "WI_data"), MdRegion, MdXyz};
  llvm::MDNode *Md = llvm::MDNode::get(Context, V);

  for (auto *BB : *this) {
    for (auto &I : *BB) {
      llvm::Metadata *V3[] = {
          llvm::MDString::get(Context, "WI_counter"),
          llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), Counter))};
      llvm::MDNode *MdCounter = llvm::MDNode::get(Context, V3);
      Counter++;
      I.setMetadata("wi", Md);
      I.setMetadata("wi_counter", MdCounter);
    }
  }
}

/**
 * Inserts a new basic block to the region, Before an old basic block in
 * the region.
 *
 * Assumes the inserted block to be Before the other block in control
 * flow, that is, there should be direct CFG edge from the block to the
 * other.
 */
void ParallelRegion::AddBlockBefore(llvm::BasicBlock *Block, llvm::BasicBlock *Before) {
  llvm::BasicBlock *OldExit = exitBB();
  ParallelRegion::iterator BeforePos = find(begin(), end(), Before);
  ParallelRegion::iterator OldExitPos = find(begin(), end(), OldExit);
  assert(BeforePos != end());

  /* The old exit node might is now pushed further, at most one position.
     Whether this is the case, depends if the node was inserted Before or
     after that node in the vector. That is, if indexof(Before) < indexof(oldExit). */
  if (BeforePos < OldExitPos)
    ++exitIndex_;

  insert(BeforePos, Block);
  /* The entryIndex_ should be still correct. In case the 'Before' block
     was an old entry node, the new one replaces it as an entry node at
     the same index and the old one gets pushed forward. */
}

void ParallelRegion::AddBlockAfter(llvm::BasicBlock *BB, llvm::BasicBlock *After) {
  llvm::BasicBlock *OldExit = exitBB();
  ParallelRegion::iterator AfterPos = find(begin(), end(), After);
  ParallelRegion::iterator OldExitPos = find(begin(), end(), OldExit);
  assert(AfterPos != end());

  /* The old exit node might be pushed further, at most one position.
     Whether this is the case, depends if the node was inserted before or
     after that node in the vector. That is, if indexof(before) < indexof(oldExit). */
  if (AfterPos < OldExitPos)
    ++exitIndex_;
  AfterPos++;
  insert(AfterPos, BB);
}

bool ParallelRegion::HasBlock(llvm::BasicBlock *BB) { return find(begin(), end(), BB) != end(); }

void ParallelRegion::SetExitBB(llvm::BasicBlock *Block) {
  for (size_t I = 0; I < size(); ++I) {
    if (at(I) == Block) {
      setExitBBIndex(I);
      return;
    }
  }
  assert(false && "The block was not found in the PRegion!");
}

} // namespace hipsycl::compiler
