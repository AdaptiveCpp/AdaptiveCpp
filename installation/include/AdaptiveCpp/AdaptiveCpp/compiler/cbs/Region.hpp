//===- hipSYCL/compiler/cbs/Region.hpp - abstract CFG region --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapatations: Merged regions
//
//===----------------------------------------------------------------------===//

#ifndef RV_REGION_H
#define RV_REGION_H

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/CFG.h>

#include <set>
#include <stack>

namespace llvm {
class BasicBlock;
class raw_ostream;
} // namespace llvm

namespace hipsycl::compiler {

class RegionImpl {
public:
  virtual ~RegionImpl() {}

  virtual bool contains(const llvm::BasicBlock *BB) const = 0;
  virtual llvm::BasicBlock &getRegionEntry() const = 0;

  virtual std::string str() const = 0;

  virtual void for_blocks(std::function<bool(const llvm::BasicBlock &block)> userFunc) const {
    auto *func = getRegionEntry().getParent();
    for (const auto &BB : *func) {
      if (contains(&BB)) {
        bool carryOn = userFunc(BB);
        if (!carryOn)
          break;
      }
    }
  }

  virtual void getEndingBlocks(llvm::SmallPtrSet<llvm::BasicBlock *, 2> &endingBlocks) const {
    assert(endingBlocks.empty());

    std::stack<llvm::BasicBlock *> blockStack;
    blockStack.push(&this->getRegionEntry());

    std::set<llvm::BasicBlock *> visitedBlocks;

    while (!blockStack.empty()) {
      // Pop the next block
      llvm::BasicBlock *block = blockStack.top();
      blockStack.pop();

      // Make sure we haven't seen it already
      if (visitedBlocks.count(block))
        continue;
      visitedBlocks.insert(block);

      // If a successor is outside the region, the region ends here.
      // Successors inside the region need to be processed recursively
      for (llvm::BasicBlock *successor : successors(block)) {
        if (this->contains(successor)) {
          blockStack.push(successor);
        } else {
          endingBlocks.insert(successor);
        }
      }
    }
  }

  virtual bool isVectorLoop() const = 0;
};

class Region {
  RegionImpl &mImpl;
  llvm::SmallPtrSet<const llvm::BasicBlock *, 32> extraBlocks;

public:
  Region(RegionImpl &mImpl);
  bool contains(const llvm::BasicBlock *BB) const;

  llvm::BasicBlock &getRegionEntry() const;
  void getEndingBlocks(llvm::SmallPtrSet<llvm::BasicBlock *, 2> &endingBlocks) const;
  void print(llvm::raw_ostream &) const {}
  std::string str() const;

  void add(const llvm::BasicBlock &extra) { extraBlocks.insert(&extra); }

  // whether the region entry is a loop header thay may contain reduction phis.
  bool isVectorLoop() const;

  // iteratively apply @userFunc to all blocks in the region
  // stop if @userFunc returns false or all blocks have been prosessed, otw carry on
  void for_blocks(std::function<bool(const llvm::BasicBlock &block)> userFunc) const;

  // iteratively apply @userFunc to all blocks in the region in reverse post-order of the CFG.
  // stop if @userFunc returns false or all blocks have been prosessed, otw carry on
  void for_blocks_rpo(std::function<bool(const llvm::BasicBlock &block)> userFunc) const;

  llvm::Function &getFunction() { return *getRegionEntry().getParent(); }
  const llvm::Function &getFunction() const { return *getRegionEntry().getParent(); }
};

// this region object captures the entire CFG of a function
class FunctionRegion final : public RegionImpl {
private:
  llvm::Function &F;
  llvm::SmallPtrSet<llvm::BasicBlock *, 16> BBs;

public:
  FunctionRegion(llvm::Function &_F, llvm::ArrayRef<llvm::BasicBlock *> BBs)
      : F(_F), BBs(BBs.begin(), BBs.end()){};
  ~FunctionRegion() {}

  bool contains(const llvm::BasicBlock *BB) const override { return BBs.contains(BB); }
  llvm::BasicBlock &getRegionEntry() const override { return F.getEntryBlock(); }
  void getEndingBlocks(llvm::SmallPtrSet<llvm::BasicBlock *, 2> &endingBlocks) const override;
  std::string str() const override;
  bool isVectorLoop() const override { return false; }
};

// This implementation realizes regions
// with a single point of entry and exit
// All block dominated by the entry and postdominated
// by the exit are contained in this region
// The region represented this way has control flow
// possibly diverge after the entry but reconverge
// at the exit
class LoopRegion final : public RegionImpl {
private:
  llvm::Loop &loop;

public:
  LoopRegion(llvm::Loop &);
  ~LoopRegion();

  bool contains(const llvm::BasicBlock *BB) const override;
  llvm::BasicBlock &getRegionEntry() const override;
  void getEndingBlocks(llvm::SmallPtrSet<llvm::BasicBlock *, 2> &endingBlocks) const override;
  std::string str() const override;
  bool isVectorLoop() const override { return true; }
};

} // namespace hipsycl::compiler

#endif // RV_REGION_H
