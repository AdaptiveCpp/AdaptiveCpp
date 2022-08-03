//===- src/compiler/cbs/AllocaSSA.cpp - state monads for allocas --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Includes, formatting
//
//===----------------------------------------------------------------------===//

#include "hipSYCL/compiler/cbs/AllocaSSA.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/Support/raw_ostream.h>
#include <vector>

using namespace llvm;

#if 1
#define IF_DEBUG_LN if (false)
#else
#define IF_DEBUG_LN if (true)
#endif

namespace hipsycl::compiler {

// static
PtrProvenance AllocaSSA::emptyProvSingle;
PtrProvenance AllocaSSA::externalProvSingle(ProvType::External);

const Value *GetAccessedPointer(const Instruction &I) {
  const auto *storeInst = dyn_cast<StoreInst>(&I);
  if (storeInst)
    return storeInst->getPointerOperand();
  else
    return cast<LoadInst>(I).getPointerOperand();
}

void AllocaSSA::computeLiveness() {
  SmallPtrSet<BasicBlock *, 2> endingBlocks;
  region.getEndingBlocks(endingBlocks);

  std::vector<BasicBlock *> stack;
  for (auto *BB : endingBlocks)
    stack.push_back(BB);

  std::set<const BasicBlock *> alreadyVisited;

  while (!stack.empty()) {
    auto &currBlock = *stack.back();
    stack.pop_back();

    IF_DEBUG_LN errs() << "live inspect " << currBlock.getName() << "\n";

    bool changed = alreadyVisited.insert(&currBlock).second;
    IF_DEBUG_LN errs() << "\tfirst: " << changed << "\n";
    auto &summary = requestBlockSummary(currBlock);

    // mark alloca live if used
    for (auto &inst : currBlock) {
      if (isa<LoadInst>(inst)) {
        const auto *ptr = GetAccessedPointer(inst);
        if (!isa<Instruction>(ptr))
          continue;
        PtrProvenance ptrProv = getProvenance(cast<Instruction>(*ptr));
        for (auto *liveAlloc : ptrProv.allocs) { // TODO support for wildcard..
          IF_DEBUG_LN errs() << "Live " << liveAlloc->getName() << "\n";
          changed |= summary.liveAllocas.insert(liveAlloc).second;
        }
      }
    }

    for (auto *pred : predecessors(&currBlock)) {
      bool predChanged = changed;

      // dont need to transfer to self
      if (pred == &currBlock)
        continue;

      // transfer liveness to predecessors
      auto &predSummary = requestBlockSummary(*pred);
      for (auto *liveAlloc : summary.liveAllocas) {
        predChanged |= predSummary.liveAllocas.insert(liveAlloc).second;
      }

      if (predChanged)
        stack.push_back(pred);
    }
  }
}

void AllocaSSA::computePointerProvenance() {
  std::vector<const BasicBlock *> worklist;
  worklist.push_back(&region.getRegionEntry());

  std::set<const BasicBlock *> seenBlocks;

  while (!worklist.empty()) {
    const auto &currBlock = *worklist.back();
    worklist.pop_back();

    // scan through block
    bool changed = seenBlocks.insert(&currBlock).second;
    for (auto &inst : currBlock) {
      const AllocaInst *allocInst = dyn_cast<AllocaInst>(&inst);
      // const PHINode * phiNode = dyn_cast<PHINode>(&inst);

      if (!inst.getType()->isPointerTy())
        continue;
      if (allocInst) {
        if (provMap.count(&inst))
          continue;
        provMap[allocInst] = PtrProvenance(allocInst);
        changed = true;

      } else if (isa<LoadInst>(inst) || isa<CallInst>(inst)) {
        // wildcard sources
        if (provMap.count(&inst))
          continue;
        provMap[&inst] = PtrProvenance(ProvType::Wildcard); // FIXME refine as necessary
        changed = true;

      } else {
        // generic transformer -> join all operand provenances
        bool instChanged = false;
        PtrProvenance oldProv = provMap[&inst];
        for (int i = 0; i < (int)inst.getNumOperands(); ++i) {
          const auto *opInst = dyn_cast<Instruction>(inst.getOperand(i));
          if (!opInst)
            continue;
          if (!provMap.count(opInst))
            continue;
          const auto &opProv = provMap[opInst];
          instChanged |= oldProv.merge(opProv);
        }

        if (instChanged) {
          provMap[&inst] = oldProv;
          changed = true;
        }
      }
    }

    if (!changed)
      continue;

    // push successors
    auto &term = *currBlock.getTerminator();
    for (int i = 0; i < (int)term.getNumSuccessors(); ++i) {
      worklist.push_back(term.getSuccessor(i));
    }
  };
}

llvm::raw_ostream &AllocaSSA::print(raw_ostream &out) const {
  out << "Pointer Provenance {\n";
  region.for_blocks_rpo([this, &out](const BasicBlock &BB) {
    bool blockPrinted = false;

    // does this block have a join?
    const auto *summary = getBlockSummary(BB);
    if (summary) { //  && !summary->getJoinSet().isBottom()) {
      out << "Block " << BB.getName() << "\n";
      if (!summary->getJoinSet().isBottom()) {
        out << "\t join ";
        summary->getJoinSet().print(out) << "\n";
      }
      if (!summary->liveAllocas.empty()) {
        out << "\t live ";
        Print(summary->liveAllocas, out) << "\n";
      }
      blockPrinted = true;
    }

    // print provenance contents
    for (const auto &inst : BB) {
      const auto &prov = getProvenance(inst);
      if (prov.isBottom())
        continue;
      if (!blockPrinted) {
        out << "Block " << BB.getName() << "\n";
        blockPrinted = true;
      }
      out << inst << " : ";
      prov.print(out) << "\n";
    }
    if (blockPrinted)
      out << "\n";
    return true;
  });
  out << "}\n";
  return out;
}

using IntSet = std::set<int>;
static IntSet GetUnwrittenArguments(const CallInst &call) {
  IntSet unwrittenArgs;
  auto *callee = call.getCalledFunction();

  // assume that all arguments are written
  if (!callee) {
    return unwrittenArgs;
  }

  // scan through declarations argument to identify unwritten pointer args
  auto itArg = callee->arg_begin();
  auto itArgEnd = callee->arg_end();
  for (int i = 0; itArg != itArgEnd; ++itArg, ++i) {
    const auto &arg = *itArg;
    // this pointer arg mau ne written
    if (arg.getType()->isPointerTy() && !arg.onlyReadsMemory())
      continue;
    // Otw, won;t write to memory through that argument
    unwrittenArgs.insert(i);
  }

  return unwrittenArgs;
}

static bool GetWrittenPointers(const Instruction &inst,
                               SmallVector<const Value *, 1> &writtenPtrs) {
  const auto *storeInst = dyn_cast<StoreInst>(&inst);
  const auto *memTransInst = dyn_cast<MemTransferInst>(&inst);
  const auto *callInst = dyn_cast<CallInst>(&inst);

  if (storeInst) {
    writtenPtrs.push_back(storeInst->getPointerOperand());
    return true;
  }

  if (memTransInst) {
    writtenPtrs.push_back(memTransInst->getDest());
    return true;
  }

  // read-only call
  if (callInst) {
    if (callInst->onlyReadsMemory()) {
      return false;
    }

    // scan through modified ptrs
    IntSet unwrittenArgIndices = GetUnwrittenArguments(*callInst);

    int i = 0;
    for (const llvm::Value *callArg : callInst->args()) {
      // can only write to pointers (well...)
      if (!callArg->getType()->isPointerTy())
        continue;

      // if we could inspect the callee, dismiss read-only args
      if (unwrittenArgIndices.count(i))
        continue;

      writtenPtrs.push_back(callArg);
      ++i;
    }

    return !writtenPtrs.empty();
  }

  return false;
}

void AllocaSSA::compute() {
  computePointerProvenance();

  computeLiveness();

  std::vector<const AllocaInst *> allocVec;

  // collect all known allocas
  for (const auto &inst : region.getRegionEntry()) {
    const auto *allocInst = dyn_cast<AllocaInst>(&inst);
    if (!allocInst)
      continue;
    allocVec.push_back(allocInst);
  }

  std::set<const BasicBlock *> worklist;
  worklist.insert(&region.getRegionEntry());

  bool keepGoing = true;
  while (keepGoing) {

    keepGoing = false;
    region.for_blocks_rpo([&](const BasicBlock &currBlock) {
      auto itItem = worklist.find(&currBlock);
      // skip this block if its not scheduled
      if (itItem == worklist.end())
        return true;
      worklist.erase(itItem);

      BlockSummary &summary = requestBlockSummary(currBlock);
      DefMap oldLastDefs = summary.lastDef;

      // TODO create LCSSA phi nodes to deal with divergent loops (LoopInfo)
      // compute provenances with disagreeing definitions from incoming values
      std::map<const AllocaInst *, Desc *> lastDefMap;
      PtrProvenance joinSet;
      for (auto *inBlock : predecessors(&currBlock)) {
        auto &inSummary = requestBlockSummary(*inBlock);

        for (auto &it : inSummary.lastDef) {
          const auto *allocInst = it.first;
          auto *lastDef = it.second;

          if (!isLive(*allocInst, currBlock))
            continue; // do not care about dead allocas

          auto itSeen = lastDefMap.find(allocInst);
          if (itSeen == lastDefMap.end()) {
            lastDefMap[allocInst] = lastDef;
          } else {
            auto *otherDef = itSeen->second;
            if (otherDef != lastDef) {
              IF_DEBUG_LN {
                errs() << "Join in " << currBlock.getName() << " defs: " << otherDef << " of "
                       << otherDef->place->getName() << " and   " << lastDef << " of "
                       << lastDef->place->getName() << " for alloca " << allocInst->getName()
                       << "\n";
              }
              joinSet.allocs.insert(allocInst);
            }
          }
        }
      }

      // update join
      bool blockChanged = summary.allocJoin.provSet.merge(joinSet);
      keepGoing |= blockChanged;

      // register join as live-in definition
      // TODO implement wildcard support
      for (auto *allocInst : joinSet.allocs) {
        lastDefMap[allocInst] = &summary.allocJoin;
      }

      // detect instructions that operate on the alloca memory states
      for (const auto &inst : currBlock) {

        SmallVector<const Value *, 1> writtenPtrs;
        if (GetWrittenPointers(inst, writtenPtrs)) {
          // join provenances
          PtrProvenance joinedProv;
          for (const auto *ptr : writtenPtrs) {
            if (!isa<Instruction>(ptr))
              continue; // FIXME for now assume that alloca pointers and other pointer sources do
                        // not mix..
            const auto &ptrProv = getProvenance(cast<Instruction>(*ptr));
            joinedProv.merge(ptrProv);
          }

          // update monadic state of all aliased allocas
          auto itEffect = instMap.find(&inst);
          Effect *memEffect = nullptr;
          if (itEffect == instMap.end()) {
            memEffect = new Effect(&inst);
            instMap[&inst] = memEffect;
            blockChanged = true;
          } else {
            memEffect = itEffect->second;
          }

          if (joinedProv.isTop()) {
            for (auto *aliasedAllocs : allocVec) {
              lastDefMap[aliasedAllocs] = memEffect;
            }
          } else {
            for (auto *aliasedAllocs : joinedProv.allocs) {
              lastDefMap[aliasedAllocs] = memEffect;
            }
          }

        } else {
          // errs() << "Skipping " << inst << "\n";
          continue;
        }
      }

      // register live out changes
      if (summary.lastDef != lastDefMap) {
        blockChanged = true;
        summary.lastDef = lastDefMap;
      }

      keepGoing |= blockChanged;
      if (!blockChanged)
        return true;

      // push successors
      auto &term = *currBlock.getTerminator();
      for (int i = 0; i < (int)term.getNumSuccessors(); ++i) {
        worklist.insert(term.getSuccessor(i));
      }

      return !worklist.empty(); // keep going
    });
  }
}

llvm::raw_ostream &Print(const AllocSet &allocs, llvm::raw_ostream &out) {
  bool first = true;
  for (const auto *alloc : allocs) {
    if (first) {
      out << "[";
      first = false;
    } else {
      out << ", ";
    }
    alloc->printAsOperand(out, true, alloc->getParent()->getParent()->getParent());
  }
  out << "]";
  return out;
}

AllocaSSA::~AllocaSSA() {
  for (auto itEff : instMap)
    delete itEff.second;
  instMap.clear();
}

} // namespace hipsycl::compiler
