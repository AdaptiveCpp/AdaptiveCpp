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

#include "hipSYCL/compiler/IRUtils.hpp"

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Dominators.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace hipsycl::compiler::utils {
llvm::Loop *updateDtAndLi(llvm::LoopInfo &LI, llvm::DominatorTree &DT, const llvm::BasicBlock *B, llvm::Function &F) {
  DT.reset();
  DT.recalculate(F);
  LI.releaseMemory();
  LI.analyze(DT);
  return LI.getLoopFor(B);
}

bool checkedInlineFunction(llvm::CallBase *CI) {
  if (CI->getCalledFunction()->isIntrinsic())
    return false;

  // needed to be valid for success log
  const auto CalleeName = CI->getCalledFunction()->getName().str();

  llvm::InlineFunctionInfo IFI;
#if LLVM_VERSION_MAJOR <= 10
  llvm::InlineResult ILR = llvm::InlineFunction(CI, IFI, nullptr);
  if (!static_cast<bool>(ILR)) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << calleeName << ">: '" << ILR.message
                 << "'\n";
#else
  llvm::InlineResult ILR = llvm::InlineFunction(*CI, IFI, nullptr);
  if (!ILR.isSuccess()) {
    llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "Failed to inline function <" << CalleeName << ">: '"
                 << ILR.getFailureReason() << "'\n";
#endif
    return false;
  }

  HIPSYCL_DEBUG_EXECUTE_INFO(llvm::outs() << HIPSYCL_DEBUG_PREFIX_INFO << "LoopSplitter inlined function <"
                                          << CalleeName << ">\n";)
  return true;
}

bool isAnnotatedParallel(llvm::Loop *TheLoop) { // from llvm for debugging.
  llvm::MDNode *DesiredLoopIdMetadata = TheLoop->getLoopID();

  if (!DesiredLoopIdMetadata)
    return false;

  llvm::MDNode *ParallelAccesses = llvm::findOptionMDForLoop(TheLoop, "llvm.loop.parallel_accesses");
  llvm::SmallPtrSet<llvm::MDNode *, 4> ParallelAccessGroups; // For scalable 'contains' check.
  if (ParallelAccesses) {
    for (const llvm::MDOperand &MD : llvm::drop_begin(ParallelAccesses->operands(), 1)) {
      llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(MD.get());
      assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
      ParallelAccessGroups.insert(AccGroup);
    }
  }

  // The loop branch contains the parallel loop metadata. In order to ensure
  // that any parallel-loop-unaware optimization pass hasn't added loop-carried
  // dependencies (thus converted the loop back to a sequential loop), check
  // that all the memory instructions in the loop belong to an access group that
  // is parallel to this loop.
  for (llvm::BasicBlock *BB : TheLoop->blocks()) {
    for (llvm::Instruction &I : *BB) {
      if (!I.mayReadOrWriteMemory())
        continue;

      if (llvm::MDNode *AccessGroup = I.getMetadata(llvm::LLVMContext::MD_access_group)) {
        auto ContainsAccessGroup = [&ParallelAccessGroups](llvm::MDNode *AG) -> bool {
          if (AG->getNumOperands() == 0) {
            assert(llvm::isValidAsAccessGroup(AG) && "Item must be an access group");
            return ParallelAccessGroups.count(AG);
          }

          for (const llvm::MDOperand &AccessListItem : AG->operands()) {
            llvm::MDNode *AccGroup = llvm::cast<llvm::MDNode>(AccessListItem.get());
            assert(llvm::isValidAsAccessGroup(AccGroup) && "List item must be an access group");
            if (ParallelAccessGroups.count(AccGroup))
              return true;
          }
          return false;
        };

        if (ContainsAccessGroup(AccessGroup))
          continue;
      }
      auto ReturnFalse = [&I]() {
        HIPSYCL_DEBUG_EXECUTE_WARNING(llvm::outs() << HIPSYCL_DEBUG_PREFIX_WARNING << "loop not parallel: ";
                                      I.print(llvm::outs()); llvm::outs() << "\n";)
        return false;
      };
      // The memory instruction can refer to the loop identifier metadata
      // directly or indirectly through another list metadata (in case of
      // nested parallel loops). The loop identifier metadata refers to
      // itself so we can check both cases with the same routine.
      llvm::MDNode *LoopIdMD = I.getMetadata(llvm::LLVMContext::MD_mem_parallel_loop_access);

      if (!LoopIdMD)
        return ReturnFalse();

      if (!llvm::is_contained(LoopIdMD->operands(), DesiredLoopIdMetadata))
        return ReturnFalse();
    }
  }
  return true;
}

void createParallelAccessesMdOrAddAccessGroup(const llvm::Function *F, llvm::Loop *const &L,
                                              llvm::MDNode *MDAccessGroup) {
  // findOptionMDForLoopID also checks if there's a loop id, so this is fine
  if (auto *ParAccesses = llvm::findOptionMDForLoopID(L->getLoopID(), "llvm.loop.parallel_accesses")) {
    llvm::SmallVector<llvm::Metadata *, 4> AccessGroups{ParAccesses->op_begin(),
                                                        ParAccesses->op_end()}; // contains .parallel_accesses
    AccessGroups.push_back(MDAccessGroup);
    auto *NewParAccesses = llvm::MDNode::get(F->getContext(), AccessGroups);

    const auto *const PIt = std::find(L->getLoopID()->op_begin(), L->getLoopID()->op_end(), ParAccesses);
    auto PIdx = std::distance(L->getLoopID()->op_begin(), PIt);
    L->getLoopID()->replaceOperandWith(PIdx, NewParAccesses);
  } else {
    auto *NewParAccesses = llvm::MDNode::get(
        F->getContext(), {llvm::MDString::get(F->getContext(), "llvm.loop.parallel_accesses"), MDAccessGroup});
    L->setLoopID(llvm::makePostTransformationMetadata(F->getContext(), L->getLoopID(), {}, {NewParAccesses}));
  }
}

void addAccessGroupMD(llvm::Instruction *I, llvm::MDNode *MDAccessGroup) {
  if (auto *PresentMD = I->getMetadata(llvm::LLVMContext::MD_access_group)) {
    llvm::SmallVector<llvm::Metadata *, 4> MDs;
    if (PresentMD->getNumOperands() == 0)
      MDs.push_back(PresentMD);
    else
      MDs.append(PresentMD->op_begin(), PresentMD->op_end());
    MDs.push_back(MDAccessGroup);
    auto *CombinedMDAccessGroup = llvm::MDNode::getDistinct(I->getContext(), MDs);
    I->setMetadata(llvm::LLVMContext::MD_access_group, CombinedMDAccessGroup);
  } else
    I->setMetadata(llvm::LLVMContext::MD_access_group, MDAccessGroup);
}

} // namespace hipsycl::compiler::utils