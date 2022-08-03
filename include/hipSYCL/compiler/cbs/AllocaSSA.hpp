//===- hipsycl/compiler/cbs/AllocaSSA.h - state monads for allocas --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Includes / Namespace, formatting
//
//===----------------------------------------------------------------------===//

#ifndef RV_ANALYSIS_ALLOCASSA_H
#define RV_ANALYSIS_ALLOCASSA_H

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/raw_ostream.h>

#include <hipSYCL/compiler/cbs/VectorizationInfo.hpp>

#include <map>

namespace hipsycl::compiler {

using AllocSet = llvm::SmallPtrSet<const llvm::AllocaInst *, 2>;

llvm::raw_ostream &Print(const AllocSet &allocs, llvm::raw_ostream &out);

// ptr provenance lattice
enum class ProvType : int32_t {
  Tracked = 0,  // only aliases with @trackedAllocs (bottom, if @trackedAllocs = \emptyset)
  External = 1, // aliases only with @trackedAllocs AND other ptr that do not alias with any allocas
  Wildcard = 2  // alises with everything (top)
};

struct PtrProvenance {
  ProvType provType; //
  AllocSet allocs;   // alias allocaInsts

  PtrProvenance() : provType(ProvType::Tracked), allocs() {}

  PtrProvenance(ProvType _provType) : provType(_provType), allocs() {}

  // single allocation ctor
  PtrProvenance(const llvm::AllocaInst *allocInst) : provType(ProvType::Tracked), allocs() {
    allocs.insert(allocInst);
  }

  // provenance lattice join
  bool merge(const PtrProvenance &O) {
    bool changed = (provType != O.provType);
    provType = std::max<ProvType>(provType, O.provType);
    if (provType == ProvType::Wildcard) {
      allocs.clear(); // explicit tracking no longed necessary
    } else {
      for (const auto *alloc : O.allocs) {
        changed |= allocs.insert(alloc).second;
      }
    }
    return changed;
  }

  bool isBottom() const { return provType == ProvType::Tracked && allocs.empty(); }
  bool isTop() const { return provType == ProvType::Wildcard; }

  llvm::raw_ostream &print(llvm::raw_ostream &out) const {
    if (provType == ProvType::Wildcard) {
      out << "*";
      return out;
    }

    Print(allocs, out);
    if (provType == ProvType::External) {
      out << "+";
    }
    return out;
  }
};

enum DescType : int32_t { JoinDesc = 0, EffectDesc = 1 };

struct Desc {
  DescType descType;
  const llvm::BasicBlock *place;

  Desc(DescType _descType, const llvm::BasicBlock *_place) : descType(_descType), place(_place) {}
};

struct Join : public Desc {
  PtrProvenance provSet; // affected allocations if this is a join of divergent, disjoint paths

  Join(const llvm::BasicBlock *_place) : Desc(DescType::JoinDesc, _place) {}
};

struct Effect : public Desc {
  const llvm::Instruction *inst;

  Effect(const llvm::Instruction *_inst)
      : Desc(DescType::EffectDesc, _inst ? _inst->getParent() : nullptr), inst(_inst) {}
};

// constructs SSA form for allocas
// associates every pointer value with the set of allocas it originates from
// the results of this analysis are used by the VectorizationAnalysis to track which allocas may
// remain uniform. this is crucial for stack allocated objects, such as stacks in data structure
// traversal codes.
class AllocaSSA {
  Region &region;
  std::map<const llvm::Instruction *, PtrProvenance> provMap;
  static PtrProvenance emptyProvSingle;    // bottom element
  static PtrProvenance externalProvSingle; // provenance object pointing to external source

  using DefMap = std::map<const llvm::AllocaInst *, Desc *>;
  struct BlockSummary {
    AllocSet liveAllocas; // computed during computeLiveness
    const llvm::BasicBlock &BB;
    Join allocJoin;
    const PtrProvenance &getJoinSet() const { return allocJoin.provSet; }

    DefMap lastDef; // live out definitions

    BlockSummary(const llvm::BasicBlock &_bb) : BB(_bb), allocJoin(&_bb) {}
  };

  std::map<const llvm::BasicBlock *, BlockSummary *> summaries;

  std::map<const llvm::Instruction *, Effect *> instMap; // owns the Effect objects

  // returns the last defining effect on @allocInst
  Desc *getLastDef(const llvm::BasicBlock &BB, const llvm::AllocaInst &allocInst) const;

  const BlockSummary *getBlockSummary(const llvm::BasicBlock &BB) const {
    auto it = summaries.find(&BB);
    if (it != summaries.end()) {
      return it->second;
    }
    return nullptr;
  }

  BlockSummary &requestBlockSummary(const llvm::BasicBlock &BB) {
    auto it = summaries.find(&BB);
    BlockSummary *summary = nullptr;
    if (it != summaries.end()) {
      summary = it->second;
    } else {
      summary = new BlockSummary(BB);
      summaries[&BB] = summary;
    }
    return *summary;
  }

  // associates every (potentially) alloca-derive pointer with its provenance
  void computePointerProvenance();

  // compute liveness per alloca
  void computeLiveness();

  bool isLive(const llvm::AllocaInst &alloca, const llvm::BasicBlock &BB) const {
    const auto *summary = getBlockSummary(BB);
    if (!summary)
      return false;
    return summary->liveAllocas.count(&alloca);
  }

public:
  // pointer provenance
  const auto &getProvenance(const llvm::Value &val) const {
    const auto *inst = llvm::dyn_cast<const llvm::Instruction>(&val);
    if (!inst)
      return externalProvSingle;

    auto it = provMap.find(inst);
    if (it == provMap.end())
      return emptyProvSingle;
    else
      return it->second;
  }

  const Join *getJoinNode(const llvm::BasicBlock &BB) const {
    const auto *summary = getBlockSummary(BB);
    if (!summary)
      return nullptr;
    return &summary->allocJoin;
  }

  AllocaSSA(Region &_region) : region(_region) {}

public:
  llvm::raw_ostream &print(llvm::raw_ostream &out) const;

  void compute();

  ~AllocaSSA();
};

} // namespace hipsycl::compiler

#endif // RV_ANALYSIS_ALLOCASSA_H
