// Implementation for VariableUniformityAnalysis function pass.
//
// Copyright (c) 2013-2019 Pekka Jääskeläinen
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

#include "hipSYCL/compiler/VariableUniformityAnalysis.hpp"
#include "hipSYCL/compiler/SplitterAnnotationAnalysis.hpp"
//#include "Kernel.h"
//#include "Workgroup.h"
//#include "WorkitemHandler.h"

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ValueSymbolTable.h"
#include "llvm/Support/CommandLine.h"

#include <iostream>
#include <sstream>

// #define DEBUG_UNIFORMITY_ANALYSIS

#ifdef DEBUG_UNIFORMITY_ANALYSIS
#include "DebugHelpers.h"
#endif

// Recursively mark the canonical induction variable PHI as uniform.
// If there's a canonical induction variable in loops, the variable
// update for each iteration should be uniform. Note: this does not yet
// imply all the work-items execute the loop same number of times!
void hipsycl::compiler::VariableUniformityInfo::markInductionVariables(llvm::Function &F, llvm::Loop &L) {

  if (llvm::PHINode *InductionVar = L.getCanonicalInductionVariable()) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
    std::cerr << "### canonical induction variable, assuming uniform:";
    inductionVar->dump();
#endif
    setUniform(&F, InductionVar);
  }
  for (llvm::Loop *Subloop : L.getSubLoops()) {
    markInductionVariables(F, *Subloop);
  }
}

/**
 * Returns true in case the value should be privatized, e.g., a copy
 * should be created for each parallel work-item.
 *
 * This is not the same as !isUniform() because of some of the allocas.
 * Specifically, the loop iteration variables are sometimes uniform,
 * that is, each work item sees the same induction variable value at every iteration,
 * but the variables should be still replicated to avoid multiple increments
 * of the same induction variable by each work-item.
 */
bool hipsycl::compiler::VariableUniformityInfo::shouldBePrivatized(llvm::Function *F, llvm::Value *Val) {
  if (!isUniform(F, Val))
    return true;

  /* Check if the value is stored in stack (is an alloca or writes to an alloca). */
  /* It should be enough to context save the initial alloca and the stores to
     make sure each work-item gets their own stack slot and they are updated.
     How the value (based on which of those allocas) is computed does not matter as
     we are deadling with uniform computation. */

  if (llvm::isa<llvm::AllocaInst>(Val))
    return true;

  if (llvm::isa<llvm::StoreInst>(Val) &&
      llvm::isa<llvm::AllocaInst>(llvm::dyn_cast<llvm::StoreInst>(Val)->getPointerOperand()))
    return true;
  return false;
}

/**
 * BB divergence analysis.
 *
 * Define:
 * Uniform BB. A basic block which is known to be executed by all or none
 * of the work-items, that is, a BB where it's known safe to add a barrier.
 *
 * Divergent/varying BB. A basic block where work-items *might* diverge.
 * That is, it cannot be proven that all work-items execute the BB.
 *
 * Propagate the information from the entry downwards (breadth first).
 * This avoids infinite recursion with loop back edges and enables
 * to keep book of the "last seen" uniform BB.
 *
 * The conditions to mark a BB 'uniform':
 *
 * a) the function entry, or
 * b) BBs that post-dominate at least one uniform BB (try the previously
 *    found one), or
 * c) BBs that are branched to directly from a uniform BB using a uniform branch.
 *    Note: This assumes the CFG is well-formed in a way that there cannot be a divergent
 *    branch to the same BB in that case.
 *
 * Otherwise, assume divergent (might not be *proven* to be one!).
 *
 */
void hipsycl::compiler::VariableUniformityInfo::analyzeBBDivergence(llvm::Function *F, llvm::BasicBlock *Bb,
                                                                    llvm::BasicBlock *previousUniformBB,
                                                                    llvm::PostDominatorTree &PDT) {

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### Analyzing BB divergence (bb=" << bb->getName().str()
            << ", prevUniform=" << previousUniformBB->getName().str() << ")" << std::endl;
#endif

  auto *Term = previousUniformBB->getTerminator();
  if (Term == NULL) {
    // this is most likely a function with a single basic block, the entry
    // node, which ends with a ret
    return;
  }

  llvm::BranchInst *BrInst = llvm::dyn_cast<llvm::BranchInst>(Term);
  llvm::SwitchInst *SwInst = llvm::dyn_cast<llvm::SwitchInst>(Term);

  if (BrInst == nullptr && SwInst == nullptr) {
    // Can only handle branches and switches for now.
    return;
  }

  // The BBs that were found uniform.
  std::vector<llvm::BasicBlock *> FoundUniforms;

  // Condition c)
  if ((BrInst && (!BrInst->isConditional() || isUniform(F, BrInst->getCondition()))) ||
      (SwInst && isUniform(F, SwInst->getCondition()))) {
    // This is a branch with a uniform condition, propagate the uniformity
    // to the BB of interest.
    for (unsigned suc = 0, end = Term->getNumSuccessors(); suc < end; ++suc) {
      llvm::BasicBlock *Successor = Term->getSuccessor(suc);
      // TODO: should we check that there are no divergent entries to this
      // BB even though if the currently checked condition is uniform?
      setUniform(F, Successor, true);
      FoundUniforms.push_back(Successor);
    }
  }

  // Condition b)
  if (FoundUniforms.size() == 0) {
    if (PDT.dominates(Bb, previousUniformBB)) {
      setUniform(F, Bb, true);
      FoundUniforms.push_back(Bb);
    }
  }

  /* Assume diverging. */
  if (!isUniformityAnalyzed(F, Bb))
    setUniform(F, Bb, false);

  for (auto *UniformBB : FoundUniforms) {

    // Propagate the Uniform BB data downwards.
    auto *NextTerm = UniformBB->getTerminator();

    for (auto *NextBB : successors(NextTerm)) {
      if (!isUniformityAnalyzed(F, NextBB)) {
        analyzeBBDivergence(F, NextBB, UniformBB, PDT);
      }
    }
  }
}

bool hipsycl::compiler::VariableUniformityInfo::isUniformityAnalyzed(llvm::Function *F, llvm::Value *V) const {
  UniformityIndex &Cache = UniformityCache_[F];
  return Cache.find(V) != Cache.end();
}

/**
 * Simple uniformity analysis that recursively analyses all the
 * operands affecting the value.
 *
 * Known uniform Values that act as "leafs" in the recursive uniformity
 * check logic:
 * a) kernel arguments
 * b) constants
 * c) OpenCL C identifiers that are constant for all work-items in a work-group
 *
 */
bool hipsycl::compiler::VariableUniformityInfo::isUniform(llvm::Function *f, llvm::Value *v) {

  UniformityIndex &Cache = UniformityCache_[f];
  UniformityIndex::const_iterator i = Cache.find(v);
  if (i != Cache.end()) {
    return (*i).second;
  }

  if (llvm::BasicBlock *bb = llvm::dyn_cast<llvm::BasicBlock>(v)) {
    if (bb == &f->getEntryBlock()) {
      setUniform(f, v, true);
      return true;
    }
  }

  if (llvm::isa<llvm::Argument>(v)) {
    setUniform(f, v, true);
    return true;
  }

  if (llvm::isa<llvm::ConstantInt>(v)) {
    setUniform(f, v, true);
    return true;
  }

  if (llvm::isa<llvm::AllocaInst>(v)) {
    /* Allocas might or might not be divergent. These are produced
       from work-item private arrays or the PHIsToAllocas. It depends
       what is written to them whether they are really divergent.

       We need to figure out if any of the stores to the alloca contain
       work-item id dependent data. Take a white listing approach that
       detects the ex-phi allocas of loop iteration variables of non-diverging
       loops.

       Currently the following case is white listed:
       a) are scalars, and
       b) are accessed only with load and stores (e.g. address not taken) from
          uniform basic blocks, and
       c) the stored data is uniform

       Because alloca data can be modified in loops and thus be dependent on
       itself, we need a bit involved mechanism to handle it. First create
       a copy of the uniformity cache, then assume the alloca itself is uniform,
       then check if all the stores to the alloca contain uniform data. If
       our initial assumption was wrong, restore the cache from the backup.
    */
    UniformityCache backupCache(UniformityCache_);
    setUniform(f, v);

    bool isUniformAlloca = true;
    llvm::Instruction *instruction = llvm::dyn_cast<llvm::AllocaInst>(v);
    for (auto *U : instruction->users()) {
      llvm::Instruction *user = llvm::cast<llvm::Instruction>(U);
      if (user == NULL)
        continue;

      llvm::StoreInst *store = llvm::dyn_cast<llvm::StoreInst>(user);
      if (store) {
        if (!isUniform(f, store->getValueOperand()) || !isUniform(f, store->getParent())) {
          if (!isUniform(f, store->getParent())) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
            std::cerr << "### alloca was written in a non-uniform BB" << std::endl;
            store->getParent()->dump();
            /* TODO: This is a problematic chicken-egg situation because the
               BB uniformity check ends up analyzing allocas in phi-removed code:
               the loop constructs refer to these allocas and at that point we
               do not yet know if the BB itself is uniform. This leads to not
               being able to detect loop iteration variables as uniform. */
#endif
          }
          isUniformAlloca = false;
          break;
        }
      } else if (llvm::isa<llvm::LoadInst>(user) || llvm::isa<llvm::BitCastInst>(user)) {
      } else {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
        std::cerr << "### alloca has a suspicious user" << std::endl;
        user->dump();
#endif
        isUniformAlloca = false;
        break;
      }
    }

    if (!isUniformAlloca) {
      // restore the old uniform data as our guess was wrong
      UniformityCache_ = backupCache;
    }
    setUniform(f, v, isUniformAlloca);

    return isUniformAlloca;
  }

  /* TODO: global memory loads are uniform in case they are accessing
     the higher scope ids (group_id_?). */
  if (llvm::isa<llvm::LoadInst>(v)) {
    llvm::LoadInst *load = llvm::dyn_cast<llvm::LoadInst>(v);
    llvm::Value *pointer = load->getPointerOperand();
    llvm::Module *M = load->getParent()->getParent()->getParent();

    if (pointer == M->getGlobalVariable("_group_id_x") || pointer == M->getGlobalVariable("_group_id_y") ||
        pointer == M->getGlobalVariable("_group_id_z") || pointer == M->getGlobalVariable("_work_dim") ||
        pointer == M->getGlobalVariable("_num_groups_x") || pointer == M->getGlobalVariable("_num_groups_y") ||
        pointer == M->getGlobalVariable("_num_groups_z") || pointer == M->getGlobalVariable("_global_offset_x") ||
        pointer == M->getGlobalVariable("_global_offset_y") || pointer == M->getGlobalVariable("_global_offset_z") ||
        pointer == M->getGlobalVariable("_local_size_x") || pointer == M->getGlobalVariable("_local_size_y") ||
        pointer == M->getGlobalVariable("_local_size_z")) {

      setUniform(f, v, true);
      return true;
    }
  }

  if (llvm::isa<llvm::PHINode>(v)) {
    /* TODO: PHINodes need control flow analysis:
       even if the values are uniform, the selected
       value depends on the preceeding basic block which
       might depend on the ID. Assume they are not uniform
       for now in general and treat the loop iteration
       variable as a special case (set externally from a LoopPass).

       TODO: PHINodes can depend (indirectly or directly) on itself in loops
       so it would need infinite recursion checking.
    */
    setUniform(f, v, false);
    return false;
  }

  llvm::Instruction *instr = llvm::dyn_cast<llvm::Instruction>(v);
  if (instr == NULL) {
    setUniform(f, v, false);
    return false;
  }

  // Atomic operations might look like uniform if only considering the operands
  // (access a global memory location of which ordering by default is not
  // constrained), but their semantics have ordering: Each work-item should get
  // their own value from that memory location.
  if (instr->isAtomic()) {
    setUniform(f, v, false);
    return false;
  }

  // not computed previously, scan all operands of the instruction
  // and figure out their uniformity recursively
  for (unsigned opr = 0; opr < instr->getNumOperands(); ++opr) {
    llvm::Value *operand = instr->getOperand(opr);
    if (!isUniform(f, operand)) {
      setUniform(f, v, false);
      return false;
    }
  }
  setUniform(f, v, true);
  return true;
}

void hipsycl::compiler::VariableUniformityInfo::setUniform(llvm::Function *f, llvm::Value *v, bool isUniform) {

  UniformityIndex &cache = UniformityCache_[f];
  cache[v] = isUniform;

#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### ";
  if (isUniform)
    std::cerr << "uniform ";
  else
    std::cerr << "varying ";

  if (llvm::isa<llvm::BasicBlock>(v)) {
    std::cerr << "BB: " << v->getName().str() << std::endl;
  } else {
    v->dump();
  }
#endif
}

bool hipsycl::compiler::VariableUniformityInfo::doFinalization(llvm::Module & /*M*/) {
  UniformityCache_.clear();
  return true;
}

hipsycl::compiler::VariableUniformityInfo::VariableUniformityInfo() {}

void hipsycl::compiler::VariableUniformityInfo::analyzeFunction(llvm::Function &F, llvm::LoopInfo &LI,
                                                                llvm::PostDominatorTree &PDT) {
#ifdef DEBUG_UNIFORMITY_ANALYSIS
  std::cerr << "### refreshing VUA" << std::endl;
  dumpCFG(F, F.getName().str() + ".vua.dot");
  F.dump();
#endif

  /* Do the actual analysis on-demand except for the basic block
     divergence analysis. */
  UniformityCache_[&F].clear();

  for (auto *L : LI) {
    markInductionVariables(F, *L);
  }

  setUniform(&F, &F.getEntryBlock());
  analyzeBBDivergence(&F, &F.getEntryBlock(), &F.getEntryBlock(), PDT);
}

char hipsycl::compiler::VariableUniformityAnalysisLegacy::ID = 0;

void hipsycl::compiler::VariableUniformityAnalysisLegacy::getAnalysisUsage(llvm::AnalysisUsage &AU) const {
  AU.addRequired<llvm::PostDominatorTreeWrapperPass>();
  AU.addPreserved<llvm::PostDominatorTreeWrapperPass>();

  AU.addRequired<llvm::LoopInfoWrapperPass>();
  AU.addPreserved<llvm::LoopInfoWrapperPass>();
  // required by LoopInfo:
  AU.addRequired<llvm::DominatorTreeWrapperPass>();
  AU.addPreserved<llvm::DominatorTreeWrapperPass>();

  AU.addRequired<SplitterAnnotationAnalysisLegacy>();
  AU.addPreserved<SplitterAnnotationAnalysisLegacy>();
}

bool hipsycl::compiler::VariableUniformityAnalysisLegacy::runOnFunction(llvm::Function &F) {
  const auto &SAA = getAnalysis<SplitterAnnotationAnalysisLegacy>().getAnnotationInfo();
  if (!SAA.isKernelFunc(&F))
    return false;

  llvm::LoopInfo &LI = getAnalysis<llvm::LoopInfoWrapperPass>().getLoopInfo();
  llvm::PostDominatorTree &PDT = getAnalysis<llvm::PostDominatorTreeWrapperPass>().getPostDomTree();

  if (!VariableUniformity_)
    VariableUniformity_ = VariableUniformityInfo{};
  VariableUniformity_->analyzeFunction(F, LI, PDT);

  return false;
}

hipsycl::compiler::VariableUniformityAnalysis::Result
hipsycl::compiler::VariableUniformityAnalysis::run(llvm::Function &F, llvm::FunctionAnalysisManager &AM) {
  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  const auto *SAA = MAM.getCachedResult<hipsycl::compiler::SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F)) {
    assert(SAA && "SAA must be cached!");
    return VariableUniformityInfo{};
  }

  auto &PDT = AM.getResult<llvm::PostDominatorTreeAnalysis>(F);
  auto &LI = AM.getResult<llvm::LoopAnalysis>(F);

  VariableUniformityInfo VariableUniformity{};
  VariableUniformity.analyzeFunction(F, LI, PDT);

  return VariableUniformity;
}
llvm::AnalysisKey hipsycl::compiler::VariableUniformityAnalysis::Key;
