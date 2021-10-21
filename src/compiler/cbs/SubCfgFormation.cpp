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
#include "hipSYCL/compiler/cbs/UniformityAnalysis.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/BasicBlockUtils.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>

namespace {
using namespace hipsycl::compiler;

static const std::array<char, 3> DimName{'x', 'y', 'z'};

llvm::Value *getLoadForGlobalVariable(llvm::Function &F, llvm::StringRef VarName) {
  auto *GV = F.getParent()->getGlobalVariable(VarName);
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *LoadI = llvm::dyn_cast<llvm::LoadInst>(&I)) {
        if (LoadI->getPointerOperand() == GV)
          return &I;
      }
    }
  }
  return nullptr;
}

std::size_t getRangeDim(llvm::Function &F) {
  auto FName = F.getName();
  // todo: fix with MS mangling
  //  llvm::Regex Rgx("7nd_itemILi([1-3])E");
  llvm::Regex Rgx("EELi([1-3])E");
  llvm::SmallVector<llvm::StringRef, 4> Matches;
  if (Rgx.match(FName, &Matches))
    return std::stoull(static_cast<std::string>(Matches[1]));
  llvm_unreachable("[SubCFG] Could not deduce kernel dimensionality!");
}

llvm::SmallVector<llvm::Value *, 3> getLocalSizeValues(llvm::Function &F, int Dim) {
  auto &DL = F.getParent()->getDataLayout();
  const auto ReqdWgSize = utils::getReqdWgSize(F);

  if (ReqdWgSize[0] == 0) {
    auto *LocalSizeArg =
        std::find_if(F.arg_begin(), F.arg_end(), [](llvm::Argument &Arg) { return Arg.getName() == "local_size"; });
    if (LocalSizeArg == F.arg_end()) {
      if (Dim == 1) {
        LocalSizeArg = std::find_if(F.arg_begin(), F.arg_end(),
                                    [](llvm::Argument &Arg) { return Arg.getName() == "local_size.coerce"; });
        if (LocalSizeArg == F.arg_end())
          llvm_unreachable("[SubCFG] Kernel has no local_size or local_size.coerce argument!");
        else
          return {LocalSizeArg};
      } else if (Dim == 2) {
        auto *LocalSizeArgX = std::find_if(F.arg_begin(), F.arg_end(),
                                           [](llvm::Argument &Arg) { return Arg.getName() == "local_size.coerce0"; });
        auto *LocalSizeArgY = std::find_if(F.arg_begin(), F.arg_end(),
                                           [](llvm::Argument &Arg) { return Arg.getName() == "local_size.coerce1"; });
        if (LocalSizeArgX == F.arg_end() || LocalSizeArgY == F.arg_end())
          llvm_unreachable("[SubCFG] Kernel has no local_size or local_size.coerce{0,1} argument!");
        else
          return {LocalSizeArgX, LocalSizeArgY};
      } else
        llvm_unreachable("[SubCFG] Kernel has no local_size argument!");
    }

    // local_size is just an array of size_t's..
    auto SizeTSize = DL.getLargestLegalIntTypeSizeInBits();

    llvm::IRBuilder Builder{F.getEntryBlock().getTerminator()};
    auto *LocalSizeBC = Builder.CreatePointerCast(
        LocalSizeArg, llvm::Type::getIntNPtrTy(F.getContext(), DL.getLargestLegalIntTypeSizeInBits()),
        "local_size.cast");

    llvm::SmallVector<llvm::Value *, 3> LocalSize;
    for (int I = 0; I < Dim; ++I) {
      auto *LocalSizeGep = Builder.CreateInBoundsGEP(LocalSizeBC, {Builder.getIntN(SizeTSize, I)},
                                                     "local_size.gep." + llvm::Twine{DimName[I]});
      LocalSize.push_back(Builder.CreateLoad(LocalSizeGep, "local_size." + llvm::Twine{DimName[I]}));
    }
    return LocalSize;
  }

  HIPSYCL_DEBUG_INFO << "[SubCFG] Kernel with constant WG size: (" << ReqdWgSize[0] << "," << ReqdWgSize[1] << ","
                     << ReqdWgSize[2] << ")\n";
  auto *SizeT = DL.getLargestLegalIntType(F.getContext());
  llvm::SmallVector<llvm::Value *, 3> LocalSize;
  for (int I = 0; I < Dim; ++I)
    LocalSize.push_back(llvm::ConstantInt::get(SizeT, ReqdWgSize[I], false));
  return LocalSize;
}

std::unique_ptr<hipsycl::compiler::RegionImpl> getRegion(llvm::Function &F, const llvm::LoopInfo &LI,
                                                         llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  if (auto *WILoop = utils::getSingleWorkItemLoop(LI))
    return std::unique_ptr<hipsycl::compiler::RegionImpl>{new hipsycl::compiler::LoopRegion(*WILoop)};
  else
    return std::unique_ptr<hipsycl::compiler::RegionImpl>{new hipsycl::compiler::FunctionRegion(F, Blocks)};
}
hipsycl::compiler::VectorizationInfo getVectorizationInfo(llvm::Function &F, hipsycl::compiler::Region &R,
                                                          llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                                                          llvm::PostDominatorTree &PDT, size_t Dim) {
  hipsycl::compiler::VectorizationInfo VecInfo{F, R};
  // seed varyingness
  if (auto *WILoop = utils::getSingleWorkItemLoop(LI)) {
    VecInfo.setPinnedShape(*WILoop->getCanonicalInductionVariable(), hipsycl::compiler::VectorShape::cont());
  } else {
    for (size_t D = 0; D < Dim - 1; ++D) {
      VecInfo.setPinnedShape(*getLoadForGlobalVariable(F, LocalIdGlobalNames[D]),
                             hipsycl::compiler::VectorShape::uni());
    }
    VecInfo.setPinnedShape(*getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1]),
                           hipsycl::compiler::VectorShape::cont());
  }

  hipsycl::compiler::VectorizationAnalysis VecAna{VecInfo, LI, DT, PDT};
  VecAna.analyze();
  return VecInfo;
}

void createLoopsAround(llvm::Function &F, llvm::BasicBlock *AfterBB, const llvm::ArrayRef<llvm::Value *> &LocalSize,
                       int EntryId, llvm::ValueToValueMapTy &VMap, llvm::SmallVector<llvm::BasicBlock *, 3> &Latches,
                       llvm::BasicBlock *&LastHeader, llvm::Value *&ContiguousIdx) {
  const auto &DL = F.getParent()->getDataLayout();
  auto *LoadBB = LastHeader;
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};

  const size_t Dim = LocalSize.size();
  llvm::SmallVector<llvm::PHINode *, 3> IndVars;
  for (int D = Dim - 1; D >= 0; --D) {
    const std::string Suffix = (llvm::Twine{DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    auto *Header = llvm::BasicBlock::Create(LastHeader->getContext(), "header." + Suffix + "b", LastHeader->getParent(),
                                            LastHeader);

    Builder.SetInsertPoint(Header, Header->getFirstInsertionPt());

    auto *WIIndVar = Builder.CreatePHI(DL.getLargestLegalIntType(F.getContext()), 2, "indvar." + Suffix);
    WIIndVar->addIncoming(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 0), &F.getEntryBlock());
    IndVars.push_back(WIIndVar);
    Builder.CreateBr(LastHeader);

    auto *Latch = llvm::BasicBlock::Create(F.getContext(), "latch." + Suffix + "b", &F);
    Builder.SetInsertPoint(Latch, Latch->getFirstInsertionPt());
    auto *IncIndVar = Builder.CreateAdd(WIIndVar, Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), 1),
                                        "addInd." + Suffix, true, false);
    WIIndVar->addIncoming(IncIndVar, Latch);

    auto *LoopCond = Builder.CreateICmpULT(IncIndVar, LocalSize[D], "exit.cond." + Suffix);
    Builder.CreateCondBr(LoopCond, Header, AfterBB);
    Latches.push_back(Latch);
    LastHeader = Header;
  }

  std::reverse(Latches.begin(), Latches.end());
  std::reverse(IndVars.begin(), IndVars.end());

  for (size_t D = 1; D < Dim; ++D) {
    Latches[D]->getTerminator()->replaceSuccessorWith(AfterBB, Latches[D - 1]);
    IndVars[D]->replaceIncomingBlockWith(&F.getEntryBlock(), IndVars[D - 1]->getParent());
  }

  auto *MDWorkItemLoop = llvm::MDNode::get(F.getContext(), {llvm::MDString::get(F.getContext(), MDKind::WorkItemLoop)});
  auto *LoopID = llvm::makePostTransformationMetadata(F.getContext(), nullptr, {}, {MDWorkItemLoop});
  Latches[Dim - 1]->getTerminator()->setMetadata("llvm.loop", LoopID);
  VMap[AfterBB] = Latches[Dim - 1];

  Builder.SetInsertPoint(IndVars[Dim - 1]->getParent(), ++IndVars[Dim - 1]->getIterator());
  llvm::Value *Idx = IndVars[0];
  for (size_t D = 1; D < Dim; ++D) {
    const std::string Suffix = (llvm::Twine{DimName[D]} + ".subcfg." + llvm::Twine{EntryId}).str();

    Idx = Builder.CreateMul(Idx, LocalSize[D], "idx.mul." + Suffix, true);
    Idx = Builder.CreateAdd(IndVars[D], Idx, "idx.add." + Suffix, true);

    VMap[getLoadForGlobalVariable(F, LocalIdGlobalNames[D])] = IndVars[D];
  }

  VMap[getLoadForGlobalVariable(F, LocalIdGlobalNames[0])] = IndVars[0];

  VMap[ContiguousIdx] = Idx;
  ContiguousIdx = Idx;
}

class SubCFG {

  using BlockVector = llvm::SmallVector<llvm::BasicBlock *, 8>;
  BlockVector Blocks_;
  BlockVector NewBlocks_;
  size_t EntryId_;
  llvm::BasicBlock *EntryBarrier_;
  llvm::SmallDenseMap<llvm::BasicBlock *, size_t> ExitIds_;
  llvm::AllocaInst *LastBarrierIdStorage_;
  llvm::Value *WIIndVar_;
  llvm::BasicBlock *EntryBB_;
  llvm::BasicBlock *ExitBB_;
  llvm::BasicBlock *LoadBB_;
  llvm::BasicBlock *PreHeader_;
  size_t Dim;

  //  void addBlock(llvm::BasicBlock *BB) { Blocks_.push_back(BB); }
  llvm::BasicBlock *createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, unsigned long> BarrierPair,
                                     llvm::BasicBlock *After, llvm::BasicBlock *WILatch);

  void loadMultiSubCfgValues(
      const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
      llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createLoadBB(llvm::ValueToValueMapTy &VMap);
  llvm::BasicBlock *createUniformLoadBB(llvm::BasicBlock *OuterMostHeader);

public:
  SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
         const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
         const SplitterAnnotationInfo &SAA, llvm::Value *IndVar, size_t Dim);

  SubCFG(const SubCFG &) = delete;
  SubCFG &operator=(const SubCFG &) = delete;

  SubCFG(SubCFG &&) = default;
  SubCFG &operator=(SubCFG &&) = default;

  BlockVector &getBlocks() noexcept { return Blocks_; }
  const BlockVector &getBlocks() const noexcept { return Blocks_; }

  BlockVector &getNewBlocks() noexcept { return NewBlocks_; }
  const BlockVector &getNewBlocks() const noexcept { return NewBlocks_; }

  size_t getEntryId() const noexcept { return EntryId_; }

  llvm::BasicBlock *getEntry() noexcept { return EntryBB_; }
  llvm::BasicBlock *getExit() noexcept { return ExitBB_; }
  llvm::BasicBlock *getLoadBB() noexcept { return LoadBB_; }
  llvm::Value *getWIIndVar() noexcept { return WIIndVar_; }

  void replicate(llvm::Loop *WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap);
  void replicate(llvm::Function &WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
                 llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                 llvm::BasicBlock *AfterBB, llvm::ArrayRef<llvm::Value *> LocalSize);

  void arrayifyMultiSubCfgValues(
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
      llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
      llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, size_t ReqdArrayElements,
      hipsycl::compiler::VectorizationInfo &VecInfo);
  void fixSingleSubCfgValues(llvm::DominatorTree &DT,
                             const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                             std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo);

  void print() const;
  void removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const;
};

llvm::BasicBlock *SubCFG::createExitWithID(llvm::detail::DenseMapPair<llvm::BasicBlock *, size_t> BarrierPair,
                                           llvm::BasicBlock *After, llvm::BasicBlock *WILatch) {
  HIPSYCL_DEBUG_INFO << "Create new exit with ID: " << BarrierPair.second << " at " << After->getName() << "\n";

  auto *Exit = llvm::BasicBlock::Create(After->getContext(),
                                        After->getName() + ".subcfg.exit" + llvm::Twine{BarrierPair.second} + "b",
                                        After->getParent(), WILatch);

  auto &DL = Exit->getParent()->getParent()->getDataLayout();
  llvm::IRBuilder Builder{Exit, Exit->getFirstInsertionPt()};
  Builder.CreateStore(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), BarrierPair.second),
                      LastBarrierIdStorage_);
  Builder.CreateBr(WILatch);

  After->getTerminator()->replaceSuccessorWith(BarrierPair.first, Exit);
  return Exit;
}

SubCFG::SubCFG(llvm::BasicBlock *EntryBarrier, llvm::AllocaInst *LastBarrierIdStorage,
               const llvm::DenseMap<llvm::BasicBlock *, size_t> &BarrierIds, const llvm::Loop *WILoop,
               const SplitterAnnotationInfo &SAA, llvm::Value *IndVar, size_t Dim)
    : LastBarrierIdStorage_(LastBarrierIdStorage), EntryId_(BarrierIds.lookup(EntryBarrier)),
      EntryBarrier_(EntryBarrier), EntryBB_(EntryBarrier->getSingleSuccessor()), LoadBB_(nullptr), WIIndVar_(IndVar),
      PreHeader_(nullptr), Dim(Dim) {
  const auto *WILatch = WILoop ? WILoop->getLoopLatch() : nullptr;

  assert(WIIndVar_ && "Must have found either IndVar or __hipsycl_local_id_{x,y,z}");

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

void addRemappedDenseMapKeys(const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &OrgInstAllocaMap,
                             const llvm::ValueToValueMapTy &VMap,
                             llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &NewInstAllocaMap) {
  for (auto &InstAllocaPair : OrgInstAllocaMap) {
    if (auto *NewInst = llvm::dyn_cast_or_null<llvm::Instruction>(VMap.lookup(InstAllocaPair.first)))
      NewInstAllocaMap.insert({NewInst, InstAllocaPair.second});
  }
}

void SubCFG::replicate(
    llvm::Loop *WILoop, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap) {
  llvm::ValueToValueMapTy VMap;
  auto *OrgWIPreHeader = WILoop->getLoopPreheader();

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
  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  LoadBB_ = createLoadBB(VMap);
  VMap[EntryBarrier_] = LoadBB_;
  PreHeader_ = createUniformLoadBB(LoadBB_);
  WIHeader->replacePhiUsesWith(OrgWIPreHeader, PreHeader_);

  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);
  VMap[utils::getWorkItemLoopBodyEntry(WILoop)] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  BlocksToRemap.push_back(WIHeader);
  BlocksToRemap.push_back(WILatch);
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  auto *BrCmpI = utils::getBrCmp(*WIHeader);
  assert(BrCmpI && "WI Header must have cmp.");
  for (auto *BrOp : BrCmpI->operand_values()) {
    if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(BrOp)) {
      auto *LatchV = Phi->getIncomingValueForBlock(WILatch);

      for (auto *U : Phi->users()) {
        if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
          if (UI->getParent() == WIHeader)
            UI->replaceUsesOfWith(Phi, LatchV);
        }
      }

      // Move PHI from Header to for body
      Phi->moveBefore(&*LoadBB_->begin());
      Phi->replaceIncomingBlockWith(WILatch, WIHeader);
      VMap[WILoop->getHeader()] = LoadBB_;
      break;
    }
  }

  // Header is now latch, so copy loop md over
  WIHeader->getTerminator()->setMetadata("llvm.loop", WILatch->getTerminator()->getMetadata("llvm.loop"));
  WILatch->getTerminator()->setMetadata("llvm.loop", nullptr);

  removeDeadPhiBlocks(BlocksToRemap);

  EntryBB_ = PreHeader_;
  ExitBB_ = WIHeader;
  WIIndVar_ = VMap[WIIndVar_];
}

void SubCFG::replicate(
    llvm::Function &F, const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap, llvm::BasicBlock *AfterBB,
    llvm::ArrayRef<llvm::Value *> LocalSize) {
  auto &DL = F.getParent()->getDataLayout();
  llvm::ValueToValueMapTy VMap;

  for (auto *BB : Blocks_) {
    auto *NewBB = llvm::CloneBasicBlock(BB, VMap, ".subcfg." + llvm::Twine{EntryId_} + "b", &F);
    VMap[BB] = NewBB;
    NewBlocks_.push_back(NewBB);
    for (auto *Succ : llvm::successors(BB)) {
      if (auto ExitIt = ExitIds_.find(Succ); ExitIt != ExitIds_.end()) {
        NewBlocks_.push_back(createExitWithID(*ExitIt, NewBB, AfterBB));
      }
    }
  }

  LoadBB_ = createLoadBB(VMap);

  VMap[EntryBarrier_] = LoadBB_;

  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  llvm::BasicBlock *LastHeader = LoadBB_;
  llvm::Value *Idx = WIIndVar_;

  createLoopsAround(F, AfterBB, LocalSize, EntryId_, VMap, Latches, LastHeader, Idx);

  PreHeader_ = createUniformLoadBB(LastHeader);
  LastHeader->replacePhiUsesWith(&F.getEntryBlock(), PreHeader_);

  print();

  addRemappedDenseMapKeys(InstAllocaMap, VMap, RemappedInstAllocaMap);
  loadMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, ContInstReplicaMap, PreHeader_, VMap);

  llvm::SmallVector<llvm::BasicBlock *, 8> BlocksToRemap{NewBlocks_.begin(), NewBlocks_.end()};
  llvm::remapInstructionsInBlocks(BlocksToRemap, VMap);

  removeDeadPhiBlocks(BlocksToRemap);

  HIPSYCL_DEBUG_INFO << "[SubCFG] Idx " << *Idx << " dummy " << *WIIndVar_ << "\n";

  EntryBB_ = PreHeader_;
  ExitBB_ = Latches[0];
  WIIndVar_ = Idx;
}
void SubCFG::removeDeadPhiBlocks(llvm::SmallVector<llvm::BasicBlock *, 8> &BlocksToRemap) const {
  for (auto *BB : BlocksToRemap) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 4> Predecessors{llvm::pred_begin(BB), llvm::pred_end(BB)};
    for (auto &I : *BB) {
      if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(&I)) {
        llvm::SmallVector<llvm::BasicBlock *, 4> IncomingBlocksToRemove;
        for (int IncomingIdx = 0; IncomingIdx < Phi->getNumIncomingValues(); ++IncomingIdx) {
          auto *IncomingBB = Phi->getIncomingBlock(IncomingIdx);
          if (!Predecessors.contains(IncomingBB))
            IncomingBlocksToRemove.push_back(IncomingBB);
        }
        for (auto *IncomingBB : IncomingBlocksToRemove) {
          HIPSYCL_DEBUG_INFO << "[SubCFG] Remove incoming block " << IncomingBB->getName() << " from PHI " << *Phi
                             << "\n";
          Phi->removeIncomingValue(IncomingBB);
          HIPSYCL_DEBUG_INFO << "[SubCFG] Removed incoming block " << IncomingBB->getName() << " from PHI " << *Phi
                             << "\n";
        }
      }
    }
  }
}

bool dontArrayifyContiguousValues(
    llvm::Instruction &I, llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::Instruction *AllocaIP, size_t ReqdArrayElements, llvm::Value *IndVar,
    hipsycl::compiler::VectorizationInfo &VecInfo) {
  if (VecInfo.isPinned(I))
    return true;

  llvm::SmallVector<llvm::Instruction *, 4> WL;
  llvm::SmallPtrSet<llvm::Instruction *, 8> UniformValues;
  llvm::SmallVector<llvm::Instruction *, 8> ContiguousInsts;
  llvm::SmallPtrSet<llvm::Value *, 8> LookedAt;
  HIPSYCL_DEBUG_INFO << "[SubCFG] IndVar: " << *IndVar << "\n";
  WL.push_back(&I);
  while (!WL.empty()) {
    auto *WLValue = WL.pop_back_val();
    if (auto *WLI = llvm::dyn_cast<llvm::Instruction>(WLValue))
      for (auto *V : WLI->operand_values()) {
        HIPSYCL_DEBUG_INFO << "[SubCFG] Considering: " << *V << "\n";

        if (V == IndVar || VecInfo.isPinned(*V))
          continue;
        // todo: fix PHIs
        if (LookedAt.contains(V))
          return false;
        LookedAt.insert(V);
        if (auto *OpI = llvm::dyn_cast<llvm::Instruction>(V)) {
          if (VecInfo.getVectorShape(*OpI).isContiguous()) {
            WL.push_back(OpI);
            ContiguousInsts.push_back(OpI);
          } else if (!UniformValues.contains(OpI))
            UniformValues.insert(OpI);
        }
      }
  }
  for (auto *UI : UniformValues) {
    HIPSYCL_DEBUG_INFO << "[SubCFG] UniValue to store: " << *UI << "\n";
    if (BaseInstAllocaMap.lookup(UI))
      continue;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Store required uniform value to single element alloca " << I << "\n";
    auto *Alloca = utils::arrayifyInstruction(AllocaIP, UI, IndVar, 1);
    BaseInstAllocaMap.insert({UI, Alloca});
    VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
  }
  ContInstReplicaMap.insert({&I, ContiguousInsts});
  return true;
}

void SubCFG::arrayifyMultiSubCfgValues(
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::ArrayRef<SubCFG> SubCFGs, llvm::Instruction *AllocaIP, size_t ReqdArrayElements,
    hipsycl::compiler::VectorizationInfo &VecInfo) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> OtherCFGBlocks;
  for (auto &Cfg : SubCFGs) {
    if (&Cfg != this)
      OtherCFGBlocks.insert(Cfg.Blocks_.begin(), Cfg.Blocks_.end());
  }

  for (auto *BB : Blocks_) {
    for (auto &I : *BB) {
      if (&I == WIIndVar_)
        continue;
      if (InstAllocaMap.lookup(&I))
        continue;
      if (utils::anyOfUsers<llvm::Instruction>(&I, [&OtherCFGBlocks, this, &I](auto *UI) {
            return UI->getParent() != I.getParent() && OtherCFGBlocks.contains(UI->getParent());
          })) {
        if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(&I))
          if (auto *Alloca = utils::getLoopStateAllocaForLoad(*LInst)) {
            InstAllocaMap.insert({&I, Alloca});
            continue;
          }
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
          if (GEP->hasMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            InstAllocaMap.insert({&I, llvm::cast<llvm::AllocaInst>(GEP->getPointerOperand())});
            continue;
          }

        auto Shape = VecInfo.getVectorShape(I);
        // if (Shape.isUniform()) {
        //     HIPSYCL_DEBUG_INFO << "[SubCFG] Value uniform, store to single element alloca " << I << "\n";
        //     auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, WIIndVar_, 1);
        //     InstAllocaMap.insert({&I, Alloca});
        //     VecInfo.setVectorShape(*Alloca, hipsycl::compiler::VectorShape::uni());
        //     continue;
        // }
        if (Shape.isContiguous()) {
          if (dontArrayifyContiguousValues(I, BaseInstAllocaMap, ContInstReplicaMap, AllocaIP, ReqdArrayElements,
                                           WIIndVar_, VecInfo)) {
            HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying " << I << "\n";
            continue;
          }
        }
        auto *Alloca = utils::arrayifyInstruction(AllocaIP, &I, WIIndVar_, ReqdArrayElements);
        InstAllocaMap.insert({&I, Alloca});
        VecInfo.setVectorShape(*Alloca, Shape);
      }
    }
  }
}

void remapInstruction(llvm::Instruction *I, llvm::ValueToValueMapTy &VMap) {
  llvm::SmallVector<llvm::Value *, 8> WL{I->value_op_begin(), I->value_op_end()};
  for (auto *V : WL) {
    if (VMap.count(V))
      I->replaceUsesOfWith(V, VMap[V]);
  }
  HIPSYCL_DEBUG_INFO << "[SubCFG] remapped Inst " << *I << "\n";
}

void SubCFG::loadMultiSubCfgValues(
    const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &InstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &BaseInstAllocaMap,
    llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> &ContInstReplicaMap,
    llvm::BasicBlock *UniformLoadBB, llvm::ValueToValueMapTy &VMap) {
  llvm::Value *NewWIIndVar = VMap[WIIndVar_];
  auto *LoadTerm = LoadBB_->getTerminator();
  auto *UniformLoadTerm = UniformLoadBB->getTerminator();
  llvm::IRBuilder Builder{LoadTerm};

  for (auto &InstAllocaPair : InstAllocaMap) {
    if (std::find(Blocks_.begin(), Blocks_.end(), InstAllocaPair.first->getParent()) == Blocks_.end()) {
      if (utils::anyOfUsers<llvm::Instruction>(InstAllocaPair.first, [this](llvm::Instruction *UI) {
            return std::find(NewBlocks_.begin(), NewBlocks_.end(), UI->getParent()) != NewBlocks_.end();
          })) {
        if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(InstAllocaPair.first))
          if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
            auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(
                Builder.CreateInBoundsGEP(GEP->getPointerOperand(), {NewWIIndVar}, GEP->getName() + "c"));
            NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
            VMap[InstAllocaPair.first] = NewGEP;
            continue;
          }
        auto *IP = LoadTerm;
        if (!InstAllocaPair.second->isArrayAllocation())
          IP = UniformLoadTerm;
        HIPSYCL_DEBUG_INFO << "[SubCFG] Load from Alloca " << *InstAllocaPair.second << " in "
                           << IP->getParent()->getName() << "\n";
        auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewWIIndVar, IP, InstAllocaPair.first->getName());
        utils::copyDgbValues(InstAllocaPair.first, Load, IP);
        VMap[InstAllocaPair.first] = Load;
      }
    }
  }

  llvm::ValueToValueMapTy UniVMap;
  UniVMap[WIIndVar_] = NewWIIndVar;
  for (size_t D = 0; D < Dim; ++D) {
    auto *Load = getLoadForGlobalVariable(*LoadBB_->getParent(), LocalIdGlobalNames[D]);
    UniVMap[Load] = VMap[Load];
  }
  for (auto &InstAllocaPair : BaseInstAllocaMap) {
    auto *IP = UniformLoadTerm;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Load base value from Alloca " << *InstAllocaPair.second << " in "
                       << IP->getParent()->getName() << "\n";
    auto *Load = utils::loadFromAlloca(InstAllocaPair.second, NewWIIndVar, IP, InstAllocaPair.first->getName());
    utils::copyDgbValues(InstAllocaPair.first, Load, IP);
    UniVMap[InstAllocaPair.first] = Load;
  }

  llvm::SmallPtrSet<llvm::Instruction *, 16> InstsToRemap;

  for (auto &InstContInstsPair : ContInstReplicaMap) {
    if (UniVMap.count(InstContInstsPair.first))
      continue;

    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction and operands of: " << *InstContInstsPair.first << " to "
                       << LoadTerm->getParent()->getName() << "\n";
    auto *IClone = InstContInstsPair.first->clone();
    IClone->insertBefore(LoadTerm);
    InstsToRemap.insert(IClone);
    UniVMap[InstContInstsPair.first] = IClone;
    if (VMap.count(InstContInstsPair.first) == 0)
      VMap[InstContInstsPair.first] = IClone;
    HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
    for (auto *Inst : InstContInstsPair.second) {
      if (UniVMap.count(Inst))
        continue;
      IClone = Inst->clone();
      IClone->insertBefore(LoadTerm);
      InstsToRemap.insert(IClone);
      UniVMap[Inst] = IClone;
      if (VMap.count(Inst) == 0)
        VMap[Inst] = IClone;
      HIPSYCL_DEBUG_INFO << "[SubCFG] Clone cont instruction: " << *IClone << "\n";
    }
  }
  for (auto *IToRemap : InstsToRemap)
    remapInstruction(IToRemap, UniVMap);
}

llvm::BasicBlock *SubCFG::createUniformLoadBB(llvm::BasicBlock *OuterMostHeader) {
  auto *LoadBB =
      llvm::BasicBlock::Create(OuterMostHeader->getContext(), "uniloadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                               OuterMostHeader->getParent(), OuterMostHeader);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(OuterMostHeader);
  return LoadBB;
}

llvm::BasicBlock *SubCFG::createLoadBB(llvm::ValueToValueMapTy &VMap) {
  auto *NewEntry = llvm::cast<llvm::BasicBlock>(static_cast<llvm::Value *>(VMap[EntryBB_]));
  auto *LoadBB = llvm::BasicBlock::Create(NewEntry->getContext(), "loadblock.subcfg." + llvm::Twine{EntryId_} + "b",
                                          NewEntry->getParent(), NewEntry);
  llvm::IRBuilder Builder{LoadBB, LoadBB->getFirstInsertionPt()};
  Builder.CreateBr(NewEntry);
  return LoadBB;
}

void SubCFG::fixSingleSubCfgValues(llvm::DominatorTree &DT,
                                   const llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> &RemappedInstAllocaMap,
                                   std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo) {

  auto *AllocaIP = LoadBB_->getParent()->getEntryBlock().getFirstNonPHIOrDbgOrLifetime();
  auto *LoadIP = LoadBB_->getTerminator();
  auto *UniLoadIP = PreHeader_->getTerminator();
  llvm::IRBuilder Builder{LoadIP};

  llvm::DenseMap<llvm::Instruction *, llvm::Instruction *> InstLoadMap;

  for (auto *BB : NewBlocks_) {
    llvm::SmallVector<llvm::Instruction *, 16> Insts{};
    std::transform(BB->begin(), BB->end(), std::back_inserter(Insts), [](auto &I) { return &I; });
    for (auto *Inst : Insts) {
      auto &I = *Inst;
      //      if (llvm::isa<llvm::PHINode>(I))
      //        continue;
      for (auto *OPV : I.operand_values()) {
        if (auto *OPI = llvm::dyn_cast<llvm::Instruction>(OPV); OPI && !DT.dominates(OPI, &I)) {
          if (auto *Phi = llvm::dyn_cast<llvm::PHINode>(Inst)) {
            bool FoundIncoming = false;
            for (auto &Incoming : Phi->incoming_values()) {
              if (OPV == Incoming.get()) {
                auto *IncomingBB = Phi->getIncomingBlock(Incoming);
                if (DT.dominates(OPI, IncomingBB->getTerminator())) {
                  FoundIncoming = true;
                  break;
                }
              }
            }
            if (FoundIncoming)
              continue;
          }
          HIPSYCL_DEBUG_WARNING << "Instruction not dominated " << I << " operand: " << *OPI << "\n";

          if (auto *Load = InstLoadMap.lookup(OPI))
            // if the already inserted Load does not dominate I, we must create another load.
            if (DT.dominates(Load, &I)) {
              I.replaceUsesOfWith(OPI, Load);
              continue;
            }

          if (auto *GEP = llvm::dyn_cast<llvm::GetElementPtrInst>(OPI))
            if (auto *MDArrayified = GEP->getMetadata(hipsycl::compiler::MDKind::Arrayified)) {
              auto *NewGEP = llvm::cast<llvm::GetElementPtrInst>(
                  Builder.CreateInBoundsGEP(GEP->getPointerOperand(), {WIIndVar_}, GEP->getName() + "c"));
              NewGEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDArrayified);
              I.replaceUsesOfWith(OPI, NewGEP);
              InstLoadMap.insert({OPI, NewGEP});
              continue;
            }

          llvm::AllocaInst *Alloca = nullptr;
          if (auto *RemAlloca = RemappedInstAllocaMap.lookup(OPI))
            Alloca = RemAlloca;
          if (auto *LInst = llvm::dyn_cast<llvm::LoadInst>(OPI))
            Alloca = utils::getLoopStateAllocaForLoad(*LInst);
          if (!Alloca) {
            //            if (VecInfo.getVectorShape(I).isUniform())
            //              Alloca = utils::arrayifyInstruction(AllocaIP, OPI, WIIndVar_, 1);
            //            else
            Alloca = utils::arrayifyInstruction(AllocaIP, OPI, WIIndVar_, ReqdArrayElements);
            VecInfo.setVectorShape(*Alloca, VecInfo.getVectorShape(I));
          }

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          // in split loop, OPI might be used multiple times, get the user, dominating this user and insert load there
          llvm::Instruction *NewIP = &I;
          for (auto *U : OPI->users()) {
            if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U); UI && DT.dominates(UI, NewIP)) {
              NewIP = UI;
            }
          }
#else
          auto *NewIP = LoadIP;
#endif
          // if (!Alloca->isArrayAllocation())
          //  NewIP = UniLoadIP;

          auto *Load = utils::loadFromAlloca(Alloca, WIIndVar_, NewIP, OPI->getName());
          utils::copyDgbValues(OPI, Load, NewIP);

#ifdef HIPSYCL_NO_PHIS_IN_SPLIT
          I.replaceUsesOfWith(OPI, Load);
          InstLoadMap.insert({OPI, Load});
#else
          const auto NumPreds = std::distance(llvm::pred_begin(BB), llvm::pred_end(BB));
          assert(NumPreds == 2 && "Only 2 preds allowed, otherwise must have been PHI already..");
          if (NumPreds > 1 && std::find(llvm::pred_begin(BB), llvm::pred_end(BB), LoadBB_) != llvm::pred_end(BB)) {
            Builder.SetInsertPoint(BB, BB->getFirstInsertionPt());
            auto *PHINode = Builder.CreatePHI(Load->getType(), NumPreds, I.getName());
            for (auto *PredBB : llvm::predecessors(BB))
              if (PredBB == LoadBB_)
                PHINode->addIncoming(Load, PredBB);
              else
                PHINode->addIncoming(OPV, PredBB);

            I.replaceUsesOfWith(OPI, PHINode);
            InstLoadMap.insert({OPI, PHINode});
          } else {
            I.replaceUsesOfWith(OPI, Load);
            InstLoadMap.insert({OPI, Load});
          }
#endif
        }
      }
    }
  }
}

llvm::BasicBlock *createUnreachableBlock(llvm::Function &F) {
  auto *Default = llvm::BasicBlock::Create(F.getContext(), "cbs.while.default", &F);
  llvm::IRBuilder Builder{Default, Default->getFirstInsertionPt()};
  Builder.CreateUnreachable();
  return Default;
}

llvm::BasicBlock *generateWhileSwitchAround(llvm::BasicBlock *PreHeader, llvm::BasicBlock *OldEntry,
                                            llvm::BasicBlock *Exit, llvm::AllocaInst *LastBarrierIdStorage,
                                            std::vector<SubCFG> &SubCFGs) {
  auto &F = *PreHeader->getParent();
  auto &M = *F.getParent();
  const auto &DL = M.getDataLayout();

  auto *WhileHeader =
      llvm::BasicBlock::Create(PreHeader->getContext(), "cbs.while.header", PreHeader->getParent(), OldEntry);
  llvm::IRBuilder Builder{WhileHeader, WhileHeader->getFirstInsertionPt()};
  auto *LastID = Builder.CreateLoad(LastBarrierIdStorage, "cbs.while.last_barr.load");
  auto *Switch = Builder.CreateSwitch(LastID, createUnreachableBlock(F), SubCFGs.size());
  for (auto &Cfg : SubCFGs) {
    Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), Cfg.getEntryId()), Cfg.getEntry());
    Cfg.getEntry()->replacePhiUsesWith(PreHeader, WhileHeader);
    Cfg.getExit()->getTerminator()->replaceSuccessorWith(Exit, WhileHeader);
  }
  Switch->addCase(Builder.getIntN(DL.getLargestLegalIntTypeSizeInBits(), ExitBarrierId), Exit);

  Builder.SetInsertPoint(PreHeader->getTerminator());
  Builder.CreateStore(llvm::ConstantInt::get(LastBarrierIdStorage->getAllocatedType(), EntryBarrierId),
                      LastBarrierIdStorage);
  PreHeader->getTerminator()->replaceSuccessorWith(OldEntry, WhileHeader);
  return WhileHeader;
}

void purgeLifetime(SubCFG &Cfg) {
  llvm::SmallVector<llvm::Instruction *, 8> ToDelete;
  for (auto *BB : Cfg.getNewBlocks())
    for (auto &I : *BB)
      if (auto *CI = llvm::dyn_cast<llvm::CallInst>(&I))
        if (CI->getCalledFunction())
          if (CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_start ||
              CI->getCalledFunction()->getIntrinsicID() == llvm::Intrinsic::lifetime_end)
            ToDelete.push_back(CI);

  for (auto *I : ToDelete)
    I->eraseFromParent();

  //  // remove dead bitcasts
  //  for (auto *BB : Cfg.getNewBlocks())
  //    llvm::SimplifyInstructionsInBlock(BB);
}

void fillUserHull(llvm::AllocaInst *Alloca, llvm::SmallVectorImpl<llvm::Instruction *> &Hull) {
  llvm::SmallVector<llvm::Instruction *, 8> WL;
  std::transform(Alloca->user_begin(), Alloca->user_end(), std::back_inserter(WL),
                 [](auto *U) { return llvm::cast<llvm::Instruction>(U); });
  llvm::SmallPtrSet<llvm::Instruction *, 32> AlreadySeen;
  while (!WL.empty()) {
    auto *I = WL.pop_back_val();
    AlreadySeen.insert(I);
    Hull.push_back(I);
    for (auto *U : I->users()) {
      if (auto *UI = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (!AlreadySeen.contains(UI))
          if (UI->mayReadOrWriteMemory() || UI->getType()->isPointerTy())
            WL.push_back(UI);
      }
    }
  }
}

bool isAllocaSubCfgInternal(llvm::AllocaInst *Alloca, const std::vector<SubCFG> &SubCfgs,
                            const llvm::DominatorTree &DT) {
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> UserBlocks;
  {
    llvm::SmallVector<llvm::Instruction *, 32> Users;
    fillUserHull(Alloca, Users);
    utils::PtrSetWrapper<decltype(UserBlocks)> Wrapper{UserBlocks};
    std::transform(Users.begin(), Users.end(), std::inserter(Wrapper, UserBlocks.end()),
                   [](auto *I) { return I->getParent(); });
  }

  for (auto &SubCfg : SubCfgs) {
    llvm::SmallPtrSet<llvm::BasicBlock *, 8> SubCfgSet{SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end()};
    if (std::any_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet](auto *BB) { return SubCfgSet.contains(BB); }) &&
        !std::all_of(UserBlocks.begin(), UserBlocks.end(), [&SubCfgSet, Alloca](auto *BB) {
          if (SubCfgSet.contains(BB)) {
            return true;
          }
          HIPSYCL_DEBUG_INFO << "[SubCFG] BB not in subcfgset: " << BB->getName() << " for alloca: ";
          HIPSYCL_DEBUG_EXECUTE_INFO(Alloca->print(llvm::outs()); llvm::outs() << "\n";)
          return false;
        }))
      return false;
  }

  return true;
}

void arrayifyAllocas(llvm::BasicBlock *EntryBlock, llvm::DominatorTree &DT, std::vector<SubCFG> &SubCfgs,
                     std::size_t ReqdArrayElements, hipsycl::compiler::VectorizationInfo &VecInfo) {
  auto *MDAlloca =
      llvm::MDNode::get(EntryBlock->getContext(), {llvm::MDString::get(EntryBlock->getContext(), "hipSYCLLoopState")});

  llvm::SmallPtrSet<llvm::BasicBlock *, 32> SubCfgsBlocks;
  for (auto &SubCfg : SubCfgs)
    SubCfgsBlocks.insert(SubCfg.getNewBlocks().begin(), SubCfg.getNewBlocks().end());

  llvm::SmallVector<llvm::AllocaInst *, 8> WL;
  for (auto &I : *EntryBlock) {
    if (auto *Alloca = llvm::dyn_cast<llvm::AllocaInst>(&I)) {
      if (Alloca->hasMetadata(hipsycl::compiler::MDKind::Arrayified))
        continue; // already arrayified
      if (utils::anyOfUsers<llvm::Instruction>(
              Alloca, [&SubCfgsBlocks](llvm::Instruction *UI) { return !SubCfgsBlocks.contains(UI->getParent()); }))
        continue;
      if (!isAllocaSubCfgInternal(Alloca, SubCfgs, DT))
        WL.push_back(Alloca);
    }
  }

  for (auto *I : WL) {
    // todo: can we somehow enable this..?
    //    if (VecInfo.getVectorShape(*I).isUniform()) {
    //      HIPSYCL_DEBUG_INFO << "[SubCFG] Not arrayifying alloca " << *I << "\n";
    //      continue;
    //    }
    llvm::IRBuilder AllocaBuilder{I};
    llvm::Type *T = I->getAllocatedType();
    if (auto *ArrSizeC = llvm::dyn_cast<llvm::ConstantInt>(I->getArraySize())) {
      auto ArrSize = ArrSizeC->getLimitedValue();
      if (ArrSize > 1) {
        T = llvm::ArrayType::get(T, ArrSize);
        HIPSYCL_DEBUG_WARNING << "Caution, alloca was array\n";
      }
    }

    auto *Alloca = AllocaBuilder.CreateAlloca(T, AllocaBuilder.getInt32(ReqdArrayElements), I->getName() + "_alloca");
    Alloca->setAlignment(llvm::Align{hipsycl::compiler::DefaultAlignment});
    Alloca->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

    for (auto &SubCfg : SubCfgs) {
      auto *GepIp = SubCfg.getLoadBB()->getFirstNonPHIOrDbgOrLifetime();

      llvm::IRBuilder LoadBuilder{GepIp};
      auto *GEP = llvm::cast<llvm::GetElementPtrInst>(
          LoadBuilder.CreateInBoundsGEP(Alloca, {SubCfg.getWIIndVar()}, I->getName() + "_gep"));
      GEP->setMetadata(hipsycl::compiler::MDKind::Arrayified, MDAlloca);

      llvm::replaceDominatedUsesWith(I, GEP, DT, SubCfg.getLoadBB());
    }
    I->eraseFromParent();
  }
}

void moveAllocasToEntry(llvm::Function &F, llvm::ArrayRef<llvm::BasicBlock *> Blocks) {
  llvm::SmallVector<llvm::AllocaInst *, 4> AllocaWL;
  for (auto *BB : Blocks)
    for (auto &I : *BB)
      if (auto *AllocaInst = llvm::dyn_cast<llvm::AllocaInst>(&I))
        AllocaWL.push_back(AllocaInst);
  for (auto *I : AllocaWL)
    if (F.getEntryBlock().size() == 1)
      I->moveBefore(F.getEntryBlock().getFirstNonPHI());
    else
      I->moveAfter(F.getEntryBlock().getFirstNonPHI());
}

void formSubCfgs(llvm::Function &F, llvm::LoopInfo &LI, llvm::DominatorTree &DT, llvm::PostDominatorTree &PDT,
                 const SplitterAnnotationInfo &SAA) {
  auto *WILoop = utils::getSingleWorkItemLoop(LI);
  //  assert(WILoop && "Must have work item loop in kernel");
  if (WILoop) {
    assert(WILoop->getCanonicalInductionVariable() && "Must have work item index");
  }

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)

  const std::size_t Dim = getRangeDim(F);
  HIPSYCL_DEBUG_INFO << "[SubCFG] Kernel is " << Dim << "-dimensional\n";

  const auto LocalSize = getLocalSizeValues(F, Dim);

  const size_t ReqdArrayElements = utils::getReqdStackElements(F);

  auto *Entry = &F.getEntryBlock();
  if (WILoop) {
    Entry = utils::getWorkItemLoopBodyEntry(WILoop);
  }

  llvm::DenseMap<llvm::BasicBlock *, size_t> Barriers;
  llvm::SmallVector<llvm::BasicBlock *, 4> ExitingBlocks;
  if (WILoop)
    ExitingBlocks.append(llvm::pred_begin(WILoop->getLoopLatch()), llvm::pred_end(WILoop->getLoopLatch()));
  else
    for (auto &BB : F)
      if (BB.getTerminator()->getNumSuccessors() == 0)
        ExitingBlocks.push_back(&BB);

  if (ExitingBlocks.empty()) {
    HIPSYCL_DEBUG_ERROR << "[SubCFG] Invalid kernel! No kernel exits!\n";
    llvm_unreachable("[SubCFG] Invalid kernel! No kernel exits!\n");
  }

  // mark exit barrier with the corresponding id:
  for (auto *BB : ExitingBlocks)
    Barriers[BB] = ExitBarrierId;
  // mark entry barrier with the corresponding id:
  Barriers[Entry] = EntryBarrierId;

  std::vector<llvm::BasicBlock *> Blocks;
  if (WILoop)
    Blocks.insert(Blocks.begin(), WILoop->block_begin(), WILoop->block_end());
  else {
    Blocks.reserve(std::distance(F.begin(), F.end()));
    std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });
  }

  // non-entry block Allocas are considered broken, move to entry.
  moveAllocasToEntry(F, Blocks);

  auto RImpl = getRegion(F, LI, Blocks);
  hipsycl::compiler::Region R{*RImpl};
  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim);

  // store all other barrier blocks with a unique id:
  for (auto *BB : Blocks)
    if (Barriers.find(BB) == Barriers.end() && utils::hasOnlyBarrier(BB, SAA))
      Barriers.insert({BB, Barriers.size()});

  const llvm::DataLayout &DL = F.getParent()->getDataLayout();
  llvm::IRBuilder Builder{F.getEntryBlock().getFirstNonPHI()};
  auto *LastBarrierIdStorage =
      Builder.CreateAlloca(DL.getLargestLegalIntType(F.getContext()), nullptr, "LastBarrierId");

  // get a common (pseudo) index value to be replaced by the actual index later
  llvm::Instruction *IndVar = nullptr;
  if (WILoop) {
    IndVar = WILoop->getCanonicalInductionVariable();
  } else {
    Builder.SetInsertPoint(F.getEntryBlock().getTerminator());
    IndVar = Builder.CreateLoad(llvm::UndefValue::get(
        llvm::PointerType::get(getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1])->getType(), 0)));
    VecInfo.setPinnedShape(*IndVar, hipsycl::compiler::VectorShape::cont());
  }

  // create subcfgs
  std::vector<SubCFG> SubCFGs;
  for (auto &BIt : Barriers) {
    HIPSYCL_DEBUG_INFO << "Create SubCFG from " << BIt.first->getName() << "(" << BIt.first << ") id: " << BIt.second
                       << "\n";
    if (BIt.second != ExitBarrierId)
      SubCFGs.emplace_back(BIt.first, LastBarrierIdStorage, Barriers, WILoop, SAA, IndVar, Dim);
  }

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> InstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> BaseInstAllocaMap;
  llvm::DenseMap<llvm::Instruction *, llvm::SmallVector<llvm::Instruction *, 8>> InstContReplicaMap;

  for (auto &Cfg : SubCFGs)
    Cfg.arrayifyMultiSubCfgValues(InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, SubCFGs,
                                  F.getEntryBlock().getFirstNonPHI(), ReqdArrayElements, VecInfo);

  llvm::BasicBlock *ExitFuncBB = nullptr;
  if (!WILoop)
    ExitFuncBB = ExitingBlocks[0];

  llvm::DenseMap<llvm::Instruction *, llvm::AllocaInst *> RemappedInstAllocaMap;
  for (auto &Cfg : SubCFGs) {
    Cfg.print();
    if (WILoop)
      Cfg.replicate(WILoop, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap);
    else
      Cfg.replicate(F, InstAllocaMap, BaseInstAllocaMap, InstContReplicaMap, RemappedInstAllocaMap, ExitFuncBB,
                    LocalSize);
    purgeLifetime(Cfg);
  }

  llvm::BasicBlock *WhileHeader = nullptr;
  if (WILoop)
    WhileHeader = generateWhileSwitchAround(WILoop->getLoopPreheader(), WILoop->getHeader(), WILoop->getExitBlock(),
                                            LastBarrierIdStorage, SubCFGs);
  else
    WhileHeader = generateWhileSwitchAround(&F.getEntryBlock(), F.getEntryBlock().getSingleSuccessor(), ExitFuncBB,
                                            LastBarrierIdStorage, SubCFGs);

  llvm::removeUnreachableBlocks(F);

  DT.recalculate(F);
  arrayifyAllocas(&F.getEntryBlock(), DT, SubCFGs, ReqdArrayElements, VecInfo);

  for (auto &Cfg : SubCFGs) {
    Cfg.fixSingleSubCfgValues(DT, RemappedInstAllocaMap, ReqdArrayElements, VecInfo);
  }

  if (!WILoop)
    IndVar->eraseFromParent();

  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG();)
  assert(!llvm::verifyFunction(F, &llvm::errs()) && "Function verification failed");

  // simplify while loop to get single latch that isn't marked as wi-loop to prevent misunderstandings.
  auto *WhileLoop = utils::updateDtAndLi(LI, DT, WhileHeader, F);
  llvm::simplifyLoop(WhileLoop, &DT, &LI, nullptr, nullptr, nullptr, false);
}

void createLoopsAroundKernel(llvm::Function &F, llvm::DominatorTree &DT, llvm::LoopInfo &LI,
                             llvm::PostDominatorTree &PDT) {
  auto *Body =
      llvm::SplitBlock(&F.getEntryBlock(), &*F.getEntryBlock().getFirstInsertionPt(), &DT, &LI, nullptr, "wibody"
#if LLVM_VERSION_MAJOR >= 13
                       ,
                       true
#endif
      );
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG());
#if LLVM_VERSION_MAJOR >= 13
  Body = Body->getSingleSuccessor();
#endif

  llvm::BasicBlock *ExitBB = nullptr;
  for (auto &BB : F) {
    if (BB.getTerminator()->getNumSuccessors() == 0) {
      ExitBB = llvm::SplitBlock(&BB, BB.getTerminator(), &DT, &LI, nullptr, "exit"
#if LLVM_VERSION_MAJOR >= 13
                                ,
                                true
#endif
      );
#if LLVM_VERSION_MAJOR >= 13
      if (Body == &BB)
        std::swap(Body, ExitBB);
      ExitBB = &BB;
#endif
      break;
    }
  }
  // after splits need to recalc
  //  PDT.recalculate(F);

  llvm::SmallVector<llvm::BasicBlock *, 8> Blocks{};
  Blocks.reserve(std::distance(F.begin(), F.end()));
  std::transform(F.begin(), F.end(), std::back_inserter(Blocks), [](auto &BB) { return &BB; });

  moveAllocasToEntry(F, Blocks);

  const auto Dim = getRangeDim(F);
  //
  //  auto RImpl = getRegion(F, LI, Blocks);
  //  hipsycl::compiler::Region R{*RImpl};
  //  auto VecInfo = getVectorizationInfo(F, R, LI, DT, PDT, Dim);

  auto LocalSize = getLocalSizeValues(F, Dim);
  llvm::ValueToValueMapTy VMap;
  llvm::SmallVector<llvm::BasicBlock *, 3> Latches;
  auto *LastHeader = Body;
  auto *Idx = getLoadForGlobalVariable(F, LocalIdGlobalNames[Dim - 1]);
  createLoopsAround(F, ExitBB, LocalSize, 0, VMap, Latches, LastHeader, Idx);

  F.getEntryBlock().getTerminator()->setSuccessor(0, LastHeader);
  llvm::remapInstructionsInBlocks(Blocks, VMap);
  for (int D = 0; D < Dim; ++D)
    if (auto *Load = llvm::cast_or_null<llvm::LoadInst>(getLoadForGlobalVariable(F, LocalIdGlobalNames[D])))
      Load->eraseFromParent();
  HIPSYCL_DEBUG_EXECUTE_VERBOSE(F.viewCFG())
}

} // namespace

namespace hipsycl::compiler {
void SubCfgFormationPassLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addRequiredTransitive<llvm::DominatorTreeWrapperPass>();
  AU.addRequiredTransitive<llvm::PostDominatorTreeWrapperPass>();
  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool SubCfgFormationPassLegacy::runOnFunction(llvm::Function &F) {
  auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();

  if (!SAA.isKernelFunc(&F))
    return false;

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = getAnalysis<llvm::DominatorTreeWrapperPass>().getDomTree();
  auto &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();
  auto &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();

  if (utils::hasBarriers(F, SAA))
    formSubCfgs(F, LI, DT, PDT, SAA);
  else if (!utils::getSingleWorkItemLoop(LI))
    createLoopsAroundKernel(F, DT, LI, PDT);

  return false;
}

char SubCfgFormationPassLegacy::ID = 0;

llvm::PreservedAnalyses SubCfgFormationPass::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  HIPSYCL_DEBUG_INFO << "[SubCFG] Form SubCFGs in " << F.getName() << "\n";

  auto &DT = AM.getResult<llvm::DominatorTreeAnalysis>(F);
  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  if (utils::hasBarriers(F, *SAA))
    formSubCfgs(F, LI, DT, PDT, *SAA);
  else if (!utils::getSingleWorkItemLoop(LI))
    createLoopsAroundKernel(F, DT, LI, PDT);

  llvm::PreservedAnalyses PA;
  PA.preserve<SplitterAnnotationAnalysis>();
  return PA;
}
} // namespace hipsycl::compiler
