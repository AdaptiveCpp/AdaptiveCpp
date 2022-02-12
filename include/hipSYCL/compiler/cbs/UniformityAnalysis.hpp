//===- hipSYCL/compiler/cbs/UniformityAnalysis.hpp - divergence analysis --*- C++ -*-===//
//
// Adapted from the RV Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adaptations: Use local or LLVM SDA depending on the LLVM version
// Remove dependency on VectorMapping and having a dedicated FunctionAnalysisManager
//
//===----------------------------------------------------------------------===//

#ifndef RV_VECTORIZATIONANALYSIS_H_
#define RV_VECTORIZATIONANALYSIS_H_

#include "AllocaSSA.hpp"
#include "VectorShape.hpp"
#include "VectorShapeTransformer.hpp"
#include "VectorizationInfo.hpp"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/GenericDomTree.h"
#include "llvm/Support/raw_ostream.h"

#if LLVM_VERSION_MAJOR < 12
#include "SyncDependenceAnalysis.hpp"
#else
#include "llvm/Analysis/SyncDependenceAnalysis.h"
#endif

#include <map>
#include <queue>
#include <string>
#include <unordered_set>

namespace llvm {
class LoopInfo;
}

namespace hipsycl::compiler {
#if LLVM_VERSION_MAJOR < 12
using SDAnalysis = pre_llvm12_compat::SyncDependenceAnalysis;
#else
using SDAnalysis = llvm::SyncDependenceAnalysis;
#endif
using InstVec = const std::vector<const llvm::Instruction *>;

class VectorizationAnalysis {
  /// In- and output
  VectorizationInfo &vecInfo;

  /// Next instructions to handle
  std::queue<const llvm::Instruction *> mWorklist;
  std::unordered_set<const llvm::Instruction *> mOnWorklist;

  const llvm::DataLayout &layout;
  const llvm::LoopInfo &LI; // Preserves LoopInfo
  const llvm::DominatorTree &DT;

  // Divergence computation:
  SDAnalysis SDA;
  //  PredicateAnalysis PredA;

  Region region;
  AllocaSSA allocaSSA;

  llvm::DenseSet<const llvm::BasicBlock *> mControlDivergentBlocks;

  bool updateTerminator(const llvm::Instruction &Term) const;

  /// update disjoin paths divergence (and push PHIs)
  // return whether @JoinBlock is a divergent loop exit from @BranchLoop
  bool propagateJoinDivergence(const llvm::BasicBlock &JoinBlock, const llvm::Loop *BranchLoop);
  void taintLoopLiveOuts(const llvm::BasicBlock &LoopHeader);

public:
  VectorizationAnalysis(VectorizationInfo &VecInfo, llvm::LoopInfo &LI, llvm::DominatorTree &DT,
                        llvm::PostDominatorTree &PDT);

  VectorizationAnalysis(const VectorizationAnalysis &) = delete;
  VectorizationAnalysis &operator=(VectorizationAnalysis) = delete;

  void analyze();
  void updateAnalysis(InstVec &updateList);

  void addInitial(const llvm::Instruction *inst, VectorShape shape);

private:
  // worklist manipulation
  // insert @inst into the worklist if its not already not the list
  bool putOnWorklist(const llvm::Instruction &inst);
  // take an element off the worklist, return nullptr if the worklist is empty
  const llvm::Instruction *takeFromWorklist();

  /// Get the shape for a value
  //  if loop carried, this is the shape observed within the loop that defines
  //  @V
  VectorShape getShape(const llvm::Value &V) const;

  // Initialize all statically known shapes (constants, arguments via argument
  // mapping, shapes set by the user)
  void init(const llvm::Function &F);

  // adjust missing shapes to undef, optimize pointer shape alignments
  void adjustValueShapes(const llvm::Function &F);

  // Run Fix-Point-Iteration after initialization
  void compute(const llvm::Function &F);

  // specialized transfer functions
  VectorShape computePHIShape(const llvm::PHINode &phi);

  // Returns true iff the shape has been changed
  bool updateShape(const llvm::Value &V, VectorShape AT);
  void analyzeDivergence(const llvm::Instruction &termInst);

  // re-schedule al phi nodes in @Block
  void pushPHINodes(const llvm::BasicBlock &Block);

  // Returns true iff all operands currently have a computed shape
  // This is essentially a negated check for bottom
  bool pushMissingOperands(const llvm::Instruction &I);

  // push all users of @V to the worklist.
  void pushUsers(const llvm::Value &V, bool IgnoreRegion = false);

  // add all instruction of \p BB to the WL that dependend on the shape of the predicate.
  // (eg functions with side effects)
  void pushPredicatedInsts(const llvm::BasicBlock &BB);

  // Control divergence propagation (RootNodeType == Loop or Instruction(~Terminator))
  // \p BranchLoop       the loop containing the divergent node
  // \p UniqueSuccessors uniqued set of immediate successors of this node
  // \p rootNode         the node handle
  // \p domBoundBlock    Immediate dominator of all UniqueSuccessors
  void propagateControlDivergence(const llvm::Loop *BranchLoop,
                                  llvm::ArrayRef<const llvm::BasicBlock *> UniqueSuccessors,
                                  const llvm::Instruction &rootNode,
                                  const llvm::BasicBlock &domBoundBlock);

  // front-ends to propagateControlDivergence..
  // .. for divergent branches (really any degree > 1 terminator)
  // propagates control divergence caused by immediate divergence in \p Term.
  void propagateBranchDivergence(const llvm::Instruction &Term);
  // .. for divergent loops
  // propagates control divergence caused by divergent loop exits of \p ExitingLoops.
  void propagateLoopDivergence(const llvm::Loop &ExitingLoop);

  // Cast undefined instruction shapes to uniform shapes
  void promoteUndefShapesToUniform(const llvm::Function &F);
};

} // namespace hipsycl::compiler

#endif