/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2019-2024 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "hipSYCL/compiler/llvm-to-backend/GlobalSizesFitInI32OptPass.hpp"

#include <llvm/IR/Instructions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Constants.h>

namespace hipsycl {
namespace compiler {


// inserts llvm.assume calls to assert that x >= RangeMin && x < RangeMax.
bool insertRangeAssumptionForBuiltinCalls(llvm::Module &M, llvm::StringRef BuiltinName,
                                          long long RangeMin, long long RangeMax, bool MaxIsLessThanEqual) {
  llvm::Function *AssumeIntrinsic = llvm::Intrinsic::getDeclaration(&M, llvm::Intrinsic::assume);
  if(!AssumeIntrinsic)
    return false;

  if(auto* F = M.getFunction(BuiltinName)) {

    auto *ReturnedIntType = llvm::dyn_cast<llvm::IntegerType>(F->getReturnType());
    if(!ReturnedIntType)
      return false;

    for(auto* U : F->users()) {
      if(auto* C = llvm::dyn_cast<llvm::CallInst>(U)) {
        auto* NextInst = C->getNextNonDebugInstruction();

        auto *GreaterEqualMin = llvm::ICmpInst::Create(
            llvm::Instruction::OtherOps::ICmp, llvm::ICmpInst::Predicate::ICMP_SGE, C,
            llvm::ConstantInt::get(M.getContext(),
                                   llvm::APInt(ReturnedIntType->getBitWidth(), RangeMin)),
            "", NextInst);
        
        llvm::ICmpInst::Predicate MaxPredicate = llvm::ICmpInst::ICMP_SLT;
        if(MaxIsLessThanEqual)
          MaxPredicate = llvm::ICmpInst::ICMP_SLE;
        auto *LesserThanMax = llvm::ICmpInst::Create(
            llvm::Instruction::OtherOps::ICmp, MaxPredicate, C,
            llvm::ConstantInt::get(M.getContext(),
                                  llvm::APInt(ReturnedIntType->getBitWidth(), RangeMax)),
            "", NextInst);
        

        llvm::SmallVector<llvm::Value*> CallArgGreaterEqualMin{GreaterEqualMin};
        llvm::SmallVector<llvm::Value*> CallArgLesserThanMax{LesserThanMax};
        llvm::CallInst::Create(llvm::FunctionCallee(AssumeIntrinsic), CallArgGreaterEqualMin,
                                            "", NextInst);
        llvm::CallInst::Create(llvm::FunctionCallee(AssumeIntrinsic), CallArgLesserThanMax,
                                            "", NextInst);
      }
    }
  }

  return true;
}

GlobalSizesFitInI32OptPass::GlobalSizesFitInI32OptPass(bool FitsInInt, int GroupSizeX,
                                                       int GroupSizeY, int GroupSizeZ)
    : GlobalSizesFitInInt{FitsInInt}, KnownGroupSizeX{GroupSizeX}, KnownGroupSizeY{GroupSizeY},
      KnownGroupSizeZ{GroupSizeZ} {}

llvm::PreservedAnalyses GlobalSizesFitInI32OptPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  static const char* IfFitsInIntBuiltinName = "__hipsycl_sscp_if_global_sizes_fit_in_int";
  if(auto* F = M.getFunction(IfFitsInIntBuiltinName)) {
    // Add definition
    if(F->size() == 0) {
      llvm::BasicBlock *BB =
          llvm::BasicBlock::Create(M.getContext(), "", F);
      llvm::ReturnInst::Create(
          M.getContext(),
          llvm::ConstantInt::get(
              M.getContext(),
              llvm::APInt(F->getReturnType()->getIntegerBitWidth(), GlobalSizesFitInInt ? 1 : 0)),
          BB);
    }
  }

  std::size_t MaxInt = std::numeric_limits<int>::max();
  // This needs to be called regardless of whether the GlobalSizesFitInInt optimization is
  // active.

  if(GlobalSizesFitInInt) {

    if (KnownGroupSizeX > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_num_groups_x", 0,
                                                MaxInt / KnownGroupSizeX, true);
    }
    if (KnownGroupSizeY > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_num_groups_y", 0,
                                                MaxInt / KnownGroupSizeY, true);
    }
    if (KnownGroupSizeZ > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_num_groups_z", 0,
                                                MaxInt / KnownGroupSizeZ, true);
    }


    if (KnownGroupSizeX > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_group_id_x", 0,
                                                MaxInt / KnownGroupSizeX);
    }
    if (KnownGroupSizeY > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_group_id_y", 0,
                                                MaxInt / KnownGroupSizeY);
    }
    if (KnownGroupSizeZ > 0) {
      insertRangeAssumptionForBuiltinCalls(M, "__hipsycl_sscp_get_group_id_z", 0,
                                                MaxInt / KnownGroupSizeZ);
    }

  }

  return llvm::PreservedAnalyses::none();
}
}
}