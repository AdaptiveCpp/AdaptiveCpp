//===- src/compiler/cbs/Region.cpp - abstract CFG region --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptiations: Merged Region definitions in a single file.
//
//===----------------------------------------------------------------------===//
//

#include "hipSYCL/compiler/cbs/Region.hpp"

#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/IR/Function.h>

using namespace llvm;

namespace hipsycl::compiler {

Region::Region(RegionImpl &Impl) : mImpl(Impl) {}

bool Region::contains(const BasicBlock *BB) const {
  if (extraBlocks.count(BB))
    return true;
  else
    return mImpl.contains(BB);
}

BasicBlock &Region::getRegionEntry() const { return mImpl.getRegionEntry(); }

std::string Region::str() const { return mImpl.str(); }

void Region::getEndingBlocks(llvm::SmallPtrSet<BasicBlock *, 2> &endingBlocks) const {
  mImpl.getEndingBlocks(endingBlocks);
}

bool Region::isVectorLoop() const { return mImpl.isVectorLoop(); }

void Region::for_blocks(std::function<bool(const BasicBlock &block)> userFunc) const {
  mImpl.for_blocks(userFunc);
  for (auto *block : extraBlocks)
    userFunc(*block);
}

void Region::for_blocks_rpo(std::function<bool(const BasicBlock &block)> userFunc) const {
  const Function &F = *getRegionEntry().getParent();
  ReversePostOrderTraversal<const Function *> RPOT(&F);

  for (auto *BB : RPOT) {
    if (mImpl.contains(BB) || extraBlocks.count(BB))
      userFunc(*BB);
  }
}

LoopRegion::LoopRegion(Loop &_loop) : loop(_loop) {}

LoopRegion::~LoopRegion() {}

bool LoopRegion::contains(const BasicBlock *BB) const { return loop.contains(BB); }

BasicBlock &LoopRegion::getRegionEntry() const { return *loop.getHeader(); }

void LoopRegion::getEndingBlocks(llvm::SmallPtrSet<BasicBlock *, 2> &endingBlocks) const {
  SmallVector<BasicBlock *, 2> endingBlocksVector;
  loop.getExitBlocks(endingBlocksVector);

  for (auto &endingBB : endingBlocksVector) {
    endingBlocks.insert(endingBB);
  }
}

std::string LoopRegion::str() const {
  auto loopHeaderName = loop.getHeader()->getName();
  return ("LoopRegion (header " + loopHeaderName + ")").str();
}

void FunctionRegion::getEndingBlocks(llvm::SmallPtrSet<BasicBlock *, 2> &endingBlocks) const {
  for (auto *BB : BBs) {
    if (BB->getTerminator()->getNumSuccessors() == 0)
      endingBlocks.insert(BB);
  }
}

std::string FunctionRegion::str() const { return ("FunctionRegion (" + F.getName() + ")").str(); }

} // namespace hipsycl::compiler
