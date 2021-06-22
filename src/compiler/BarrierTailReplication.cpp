// LLVM function pass to replicate barrier tails (successors to barriers).
//
// Copyright (c) 2011 Universidad Rey Juan Carlos and
//               2012-2019 Pekka Jääskeläinen
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

#include <algorithm>

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/BarrierTailReplication.hpp"
#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"

#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace {

bool blockHasBarrier(const llvm::BasicBlock *BB, const hipsycl::compiler::SplitterAnnotationInfo &SAA) {
  for (const auto &I : *BB) {
    if (const auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
      if (CI->getCalledFunction() && SAA.isSplitterFunc(CI->getCalledFunction()))
        return true;
  }

  return false;
}

class BarrierTailReplication {

public:
  BarrierTailReplication(llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                         const hipsycl::compiler::SplitterAnnotationInfo &SAA)
      : DT_(DT), LI_(LI), SAA_(SAA) {}

  bool processFunction(llvm::Function &F);

private:
  typedef std::set<llvm::BasicBlock *> BasicBlockSet;
  typedef std::vector<llvm::BasicBlock *> BasicBlockVector;
  typedef std::map<llvm::Value *, llvm::Value *> ValueValueMap;

  llvm::DominatorTree &DT_;
  llvm::LoopInfo &LI_;
  const hipsycl::compiler::SplitterAnnotationInfo &SAA_;

  bool findBarriersDfs(llvm::BasicBlock *BB, BasicBlockSet &ProcessedBBs);
  bool replicateJoinedSubgraphs(llvm::BasicBlock *Dominator, llvm::BasicBlock *SubgraphEntry,
                                BasicBlockSet &ProcessedBBs);

  llvm::BasicBlock *replicateSubgraph(llvm::BasicBlock *Entry, llvm::Function *F);
  void findSubgraph(BasicBlockVector &Subgraph, llvm::BasicBlock *Entry);
  void replicateBasicBlocks(BasicBlockVector &NewGraph, llvm::ValueToValueMapTy &ReferenceMap, BasicBlockVector &Graph,
                            llvm::Function *F);
  void updateReferences(const BasicBlockVector &Graph, llvm::ValueToValueMapTy &ReferenceMap);

  bool cleanupPHIs(llvm::BasicBlock *BB);
};

//#define DEBUG_BARRIER_REPL

bool BarrierTailReplication::processFunction(llvm::Function &F) {
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### BTR on " << F.getName().str() << std::endl;
#endif

  BasicBlockSet ProcessedBbs;

  bool Changed = findBarriersDfs(&F.getEntryBlock(), ProcessedBbs);
  /* The created tails might contain PHI nodes with operands
     referring to the non-predecessor (split point) BB.
     These must be cleaned to avoid breakage later on.
   */
  for (auto &BB : F) {
    Changed |= cleanupPHIs(&BB);
  }
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  return Changed;
}

// Recursively (depth-first) look for barriers in all possible
// execution paths starting on entry, replicating the barrier
// successors to ensure there is a separate function exit BB
// for each combination of traversed barriers. The set
// ProcessedBBs stores the
bool BarrierTailReplication::findBarriersDfs(llvm::BasicBlock *BB, BasicBlockSet &ProcessedBBs) {
  bool Changed = false;

  // Check if we already visited this BB (to avoid
  // infinite recursion in case of unbarriered loops).
  if (ProcessedBBs.count(BB) != 0)
    return Changed;

  ProcessedBBs.insert(BB);

  if (blockHasBarrier(BB, SAA_)) {
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### block " << BB->getName().str() << " has barrier, RJS" << std::endl;
#endif
    BasicBlockSet ProcessedBbsRjs;
    Changed = replicateJoinedSubgraphs(BB, BB, ProcessedBbsRjs);
  }

  auto *T = BB->getTerminator();

  // Find barriers in the successors (depth first).
  for (unsigned I = 0, E = T->getNumSuccessors(); I != E; ++I)
    Changed |= findBarriersDfs(T->getSuccessor(I), ProcessedBBs);

  return Changed;
}

// Only replicate those parts of the subgraph that are not
// dominated by a (barrier) basic block, to avoid excesive
// (and confusing) code replication.
bool BarrierTailReplication::replicateJoinedSubgraphs(llvm::BasicBlock *Dominator, llvm::BasicBlock *SubgraphEntry,
                                                      BasicBlockSet &ProcessedBbs) {
  bool Changed = false;

  assert(DT_.dominates(Dominator, SubgraphEntry));

  llvm::Function *F = Dominator->getParent();

  auto *T = SubgraphEntry->getTerminator();
  for (int SucIdx = 0, NumSuc = T->getNumSuccessors(); SucIdx != NumSuc; ++SucIdx) {
    llvm::BasicBlock *BB = T->getSuccessor(SucIdx);
#ifdef DEBUG_BARRIER_REPL
    std::cerr << "### traversing from " << subgraph_entry->getName().str() << " to " << BB->getName().str()
              << std::endl;
#endif

    // Check if we already handled this BB and all its branches.
    if (ProcessedBbs.count(BB) != 0) {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### already processed " << std::endl;
#endif
      continue;
    }

    const bool IsBackedge = DT_.dominates(BB, SubgraphEntry);
    if (IsBackedge) {
      // This is a loop backedge. Do not find subgraphs across
      // those.
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### a loop backedge, skipping" << std::endl;
#endif
      continue;
    }
    if (DT_.dominates(Dominator, BB)) {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### " << dominator->getName().str() << " dominates " << BB->getName().str() << std::endl;
#endif
      Changed |= replicateJoinedSubgraphs(Dominator, BB, ProcessedBbs);
    } else {
#ifdef DEBUG_BARRIER_REPL
      std::cerr << "### " << dominator->getName().str() << " does not dominate " << BB->getName().str()
                << " replicating " << std::endl;
#endif
      llvm::BasicBlock *ReplicatedSubgraphEntry = replicateSubgraph(BB, F);
      T->setSuccessor(SucIdx, ReplicatedSubgraphEntry);
      Changed = true;
    }

    if (Changed) {
      // We have modified the function. Possibly created new loops.
      // Update analysis passes.
      hipsycl::compiler::utils::updateDtAndLi(LI_, DT_, nullptr, *F);
    }
  }
  ProcessedBbs.insert(SubgraphEntry);
  return Changed;
}

// Removes phi elements for which there are no successors (anymore).
bool BarrierTailReplication::cleanupPHIs(llvm::BasicBlock *BB) {

  bool Changed = false;
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### CleanupPHIs for BB:" << std::endl;
  BB->dump();
#endif

  for (auto BI = BB->begin(), BE = BB->end(); BI != BE;) {
    auto *PN = llvm::dyn_cast<llvm::PHINode>(&*BI);
    if (PN == NULL)
      break;

    bool PHIRemoved = false;
    for (unsigned I = 0, E = PN->getNumIncomingValues(); I < E; ++I) {
      bool IsSuccessor = false;
      // find if the predecessor branches to this one (anymore)
      for (unsigned S = 0, Se = PN->getIncomingBlock(I)->getTerminator()->getNumSuccessors(); S < Se; ++S) {
        if (PN->getIncomingBlock(I)->getTerminator()->getSuccessor(S) == BB) {
          IsSuccessor = true;
          break;
        }
      }
      if (!IsSuccessor) {
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "removing incoming value " << i << " from PHINode:" << std::endl;
        PN->dump();
#endif
        PN->removeIncomingValue(I, true);
#ifdef DEBUG_BARRIER_REPL
        std::cerr << "now:" << std::endl;
        PN->dump();
#endif
        Changed = true;
        E--;
        if (E == 0) {
          PHIRemoved = true;
          break;
        }
        I = 0;
        continue;
      }
    }
    if (PHIRemoved)
      BI = BB->begin();
    else
      BI++;
  }
  return Changed;
}

llvm::BasicBlock *BarrierTailReplication::replicateSubgraph(llvm::BasicBlock *Entry, llvm::Function *F) {
  // Find all basic blocks to replicate.
  BasicBlockVector Subgraph;
  findSubgraph(Subgraph, Entry);

  // Replicate subgraph maintaining control flow.
  BasicBlockVector V;

  llvm::ValueToValueMapTy VMap;
  replicateBasicBlocks(V, VMap, Subgraph, F);
  updateReferences(V, VMap);

  // Return entry block of replicated subgraph.
  return llvm::cast<llvm::BasicBlock>(VMap[Entry]);
}

void BarrierTailReplication::findSubgraph(BasicBlockVector &Subgraph, llvm::BasicBlock *Entry) {
  // The subgraph can have internal branches (join points)
  // avoid replicating these parts multiple times within the
  // same tail.
  if (std::count(Subgraph.begin(), Subgraph.end(), Entry) > 0)
    return;

  Subgraph.push_back(Entry);

  auto *T = Entry->getTerminator();
  for (auto *Successor : llvm::successors(T)) {
    const bool IsBackedge = DT_.dominates(Successor, Entry);
    if (IsBackedge)
      continue;
    findSubgraph(Subgraph, Successor);
  }
}

void BarrierTailReplication::replicateBasicBlocks(BasicBlockVector &NewGraph, llvm::ValueToValueMapTy &ReferenceMap,
                                                  BasicBlockVector &Graph, llvm::Function *F) {
#ifdef DEBUG_BARRIER_REPL
  std::cerr << "### ReplicateBasicBlocks: " << std::endl;
#endif
  for (auto *BB : Graph) {
    llvm::BasicBlock *NewB = llvm::BasicBlock::Create(BB->getContext(), BB->getName() + ".btr", F);
    ReferenceMap.insert(std::make_pair(BB, NewB));
    NewGraph.push_back(NewB);

#ifdef DEBUG_BARRIER_REPL
    std::cerr << "Replicated BB: " << new_b->getName().str() << std::endl;
#endif

    for (const auto &I : *BB) {
      llvm::Instruction *IClone = I.clone();
      ReferenceMap.insert(std::make_pair(&I, IClone));
      NewB->getInstList().push_back(IClone);
    }

    // Add predicates to PHINodes of basic blocks the replicated
    // block jumps to (backedges).
    auto *T = NewB->getTerminator();
    for (auto *Successor : successors(T)) {
      if (std::count(Graph.begin(), Graph.end(), Successor) == 0) {
        // Successor is not in the Graph, possible backedge.
        for (auto &SI : *Successor) {
          auto *PHI = llvm::dyn_cast<llvm::PHINode>(&SI);
          if (PHI == NULL)
            break; // All PHINodes already checked.

          // Get value for original incoming edge and add new predicate.
          llvm::Value *V = PHI->getIncomingValueForBlock(BB);
          llvm::Value *NewV = ReferenceMap.find(V) == ReferenceMap.end() ? NULL : ReferenceMap[V];

          if (NewV == NULL) {
            /* This case can happen at least when replicating a latch
               block in a b-loop. The value produced might be from a common
               path before the replicated part. Then just use the original value.*/
            NewV = V;
#if 0
            std::cerr << "### could not find a replacement block for PHI node ("
                      << b->getName().str() << ")" << std::endl;
            PHI->dump();
            v->dump();
            f->viewCFG();
            assert (0);
#endif
          }
          PHI->addIncoming(NewV, NewB);
        }
      }
    }
  }

#ifdef DEBUG_BARRIER_REPL
  std::cerr << std::endl;
#endif
}

void BarrierTailReplication::updateReferences(const BasicBlockVector &Graph, llvm::ValueToValueMapTy &ReferenceMap) {
  for (auto *BB : Graph) {
    for (auto &I : *BB) {
      RemapInstruction(&I, ReferenceMap, llvm::RF_IgnoreMissingLocals | llvm::RF_NoModuleLevelChanges);
    }
  }
}
} // namespace

void hipsycl::compiler::BarrierTailReplicationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();

  AU.addPreserved<VariableUniformityAnalysisLegacy>();
}

bool hipsycl::compiler::BarrierTailReplicationPassLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<hipsycl::compiler::SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  BarrierTailReplication BTR{DT, LI, SAA};

  return BTR.processFunction(F);
}

char hipsycl::compiler::BarrierTailReplicationPassLegacy::ID = 0;

llvm::PreservedAnalyses hipsycl::compiler::BarrierTailReplicationPass::run(llvm::Function &F,
                                                                           llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F)) {
    return llvm::PreservedAnalyses::all();
  }

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  BarrierTailReplication BTR{DT, LI, *SAA};

  if (!BTR.processFunction(F))
    return llvm::PreservedAnalyses::all();

  llvm::PreservedAnalyses PA;
  PA.preserve<VariableUniformityAnalysis>();
  PA.preserve<SplitterAnnotationAnalysis>();
  PA.preserve<llvm::LoopAnalysis>();
  PA.preserve<llvm::DominatorTreeAnalysis>();
  return PA;
}
