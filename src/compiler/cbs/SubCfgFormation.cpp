/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2021 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/cbs/SubCfgFormation.hpp"

#include "hipSYCL/compiler/IRUtils.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace {
using namespace hipsycl::compiler;

class SubCFG {

  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::Value *WIIndVar_;

  //  void addBlock(llvm::BasicBlock *BB) { Blocks_.push_back(BB); }
  llvm::BasicBlock *createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, unsigned long> BarrierPair,
                                     llvm::BasicBlock *After, llvm::BasicBlock *WILatch);

  void loadMultiSubCfgValues(const llvm::Loop *WILoop,
                             const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                             llvm::ValueToValueMapTy &VMap, llvm::BasicBlock *WIHeader);

public:
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
         const SplitterAnnotationInfo &SAA);

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  void replicate(llvm::Loop *WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap);
  void cleanupOldBlocks();

  void arrayifyMutliSubCfgValues(llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                                 llvm::SmallPtrSetImpl<const llvm::BasicBlock *> &WILoopBBs,
                                 llvm::Instruction *AllocaIP);

  void print() const;
};

llvm::BasicBlock *SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                                           llvm::BasicBlock *After, llvm::BasicBlock *WILatch) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at " << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" + llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), WILatch);

  llvm::DataLayout DL{Exit->getParent()->getParent()};
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(WILatch);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
               const SplitterAnnotationInfo &SAA)
    : LastBarrierIdStorage_(LastBarrierIdStorage), EntryId_(BarrierIds.lookup(EntryBarrier)) {
  const auto *WILatch = WILoop->getLoopLatch();
  WIIndVar_ = WILoop->getCanonicalInductionVariable();

  llvm::SmallVector<llvm::BasicBlock *, 4> WL{EntryBarrier};
  while (!WL.empty()) {
    auto *BB = WL.pop_back_val();

    llvm::SmallVector<llvm::BasicBlock *, 2> Succs{llvm::succ_begin(BB), llvm::succ_end(BB)};
    for (auto *Succ : Succs) {
      if (WILatch == Succ || std::find(Blocks_.begin(), Blocks_.end(), Succ) != Blocks_.end())
        continue;

      if (!utils::hasOnlyBarrier(Succ, SAA)) {
        WL.push_back(Succ);
        Blocks_.push_back(Succ);
      } else {
        size_t BId = BarrierIds.lookup(Succ);
        assert(BId != 0 && "Exit barrier block not found in map");
        ExitIds_.insert({Succ, BId});
      }
    }
  }
}

void SubCFG::print() const {
  HIPSYCL_DEBUG_INFO << "SubCFG entry barrier: " << EntryId_ << "\n";
  HIPSYCL_DEBUG_INFO << "SubCFG block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB : Blocks_) { llvm::outs() << BB->getName() << ", "; } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG exits: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto ExitIt
                                  : ExitIds_) {
    llvm::outs() << ExitIt.first->getName() << " (" << ExitIt.second << "), ";
  } llvm::outs() << "\n";)
  HIPSYCL_DEBUG_INFO << "SubCFG new block names: ";
  HIPSYCL_DEBUG_EXECUTE_INFO(for (auto *BB
                                  : NewBlocks_) {
    llvm::outs() << BB->getName() << ", ";
  } llvm::outs() << "\n";)
}

void SubCFG::replicate(llvm::Loop *WILoop,
                       const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap) {
  llvm::ValueToValueMapTy VMap;
  auto *WIHeader = llvm::CloneBasicBlock(WILoop->getHeader(), VMap, ".subcfg." + llvm::Twine{EntryId_} + "b",
                                         WILoop->getHeader()->getParent());
  auto *WILatch = llvm::CloneBasicBlock(WILoop->getLoopLatch(), VMap, ".subcfg." + llvm::Twine{EntryId_} + "b",
                                        WIHeader->getParent());
  VMap[WILoop->getHeader()] = WIHeader;
  VMap[WILoop->getLoopLatch()] = WILatch;

  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", WIHeader->getParent());
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, WILatch));
      }
    }
  }

  loadMultiSubCfgValues(WILoop, InstAllocaMap, VMap, WIHeader);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  BlocksToRemap.push_back(WIHeader);
  BlocksToRemap.push_back(WILatch);
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);
}

void SubCFG::cleanupOldBlocks() {
  for (auto *BB : Blocks_) {
    BB->replaceAllUsesWith(NewBlocks_.front());
    BB->eraseFromParent();
  }
  Blocks_.clear();
}

void SubCFG::arrayifyMutliSubCfgValues(llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                                       llvm::SmallPtrSetImpl<const llvm::BasicBlock *> &WILoopBBs,
                                       llvm::Instruction *AllocaIP) {
  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&WILoopBBs, this](auto *UI) {
            return std::find(Blocks_.begin(), Blocks_.end(), UI->getParent()) == Blocks_.end() &&
                   WILoopBBs.find(UI->getParent()) != WILoopBBs.end();
          }))
        InstAllocaMap.insert({&I, utils::arrayifyInstruction(AllocaIP, &I, WIIndVar_)});
    }
  }
}

void SubCFG::loadMultiSubCfgValues(const llvm::Loop *WILoop,
                                   const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                                   llvm::ValueToValueMapTy &VMap, llvm::BasicBlock *WIHeader) {
  llvm::Value *NewWIIndVar = VMap[WIIndVar_];
  auto *LoadBB = llvm::BasicBlock::Create(WIHeader->getContext(), "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          WIHeader->getParent(), NewBlocks_.front());
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  auto *LoadTerm = Builder.CreateBr(NewBlocks_.front());

  for (auto &InstAllocaPair : InstAllocaMap) {
    if (utils::anyOfUsers<llvm::Instruction>(InstAllocaPair.first, [this](auto *UI) {
          return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) != NewBlocks_.end();
        }))
      VMap[InstAllocaPair.first] =
          utils::loadFromAlloca(InstAllocaPair.second, NewWIIndVar, LoadTerm, InstAllocaPair.first->getName());
  }
  auto *OldWIEntry = utils::getWorkItemLoopBodyEntry(WILoop);
  VMap[OldWIEntry] = LoadBB;
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT, const SplitterAnnotationInfo &SAA) {
  auto *WILoop = utils::getSingleWorkItemLoop(LI);
  assert(WILoop && "Must have work item loop in kernel");
  F.viewCFG();
  llvm::PHINode *WIIndVar = WILoop->getCanonicalInductionVariable();
  assert(WIIndVar && "Must have work item index");

  auto *WIEntry = utils::getWorkItemLoopBodyEntry(WILoop);
  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;

  // mark exit barrier with the corresponding id:
  for (auto *BB : llvm::predecessors(WILoop->getLoopLatch()))
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[WIEntry] = EntryBarrierId;

  // store all other barrier blocks with a unique id:
  for (auto *BB : WILoop->blocks())
    if (Barriers.find(BB) == Barriers.end() && utils::hasOnlyBarrier(BB, SAA))
      Barriers.insert({BB, Barriers.size()});

  utils::arrayifyAllocas(&F.getEntryBlock(), *WILoop, WIIndVar, DT);

  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  for (auto &BIt : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << BIt.first->getName() << "(" << BIt.first << ") id: " << BIt.second
                       << "\n";
    if (BIt.second != ExitBarrierId) {
      SubCFGs.emplace_back(BIt.first, LastBarrierIdStorage, Barriers, WILoop, SAA);
      SubCFGs.back().arrayifyMutliSubCfgValues(InstAllocaMap, WILoop->getBlocksSet(),
                                               F.getEntryBlock().getFirstNonPHI());
    }
  }

  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    Cfg.replicate(WILoop, InstAllocaMap);
  }

  // todo: remove original structure
  //  F.viewCFG();
  //  for (auto &Cfg : SubCFGs) {
  //    Cfg.cleanupOldBlocks();
  //  }
  F.viewCFG();
}
} // namespace

namespace hipsycl::compiler {
void SubCfgFormationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  //  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequiredTransitive<llvm::DominatorTreeWrapperPass>();
  //  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool SubCfgFormationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F) || !utils::hasBarriers(F, SAA))
    return false;

  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();

  formSubCfgs(F, LI, DT, SAA);
  return false;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F) || !utils::hasBarriers(F, *SAA))
    return llvm::PreservedAnalyses::all();

  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);
  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  formSubCfgs(F, LI, DT, *SAA);

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
