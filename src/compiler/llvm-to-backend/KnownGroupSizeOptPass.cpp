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

#include "hipSYCL/compiler/llvm-to-backend/KnownGroupSizeOptPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/GlobalSizesFitInI32OptPass.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>


namespace hipsycl {
namespace compiler {

namespace {


bool applyKnownGroupSize(llvm::Module &M, int KnownGroupSize,
                         llvm::StringRef GetGroupSizeBuiltinName,
                         llvm::StringRef GetLocalIdBuiltinName) {

  // First create replacement functions for GetGroupSizeBuiltinName
  // which directly return the known group size, and replace all
  // uses.
  if(auto* GetGroupSizeF = M.getFunction(GetGroupSizeBuiltinName)) {
    std::string NewFunctionName = std::string{GetGroupSizeBuiltinName}+"_known_size";

    auto *NewGetGroupSizeF = llvm::dyn_cast<llvm::Function>(
        M.getOrInsertFunction(NewFunctionName, GetGroupSizeF->getFunctionType(),
                              GetGroupSizeF->getAttributes())
            .getCallee());
    if(!NewGetGroupSizeF)
      return false;

    if(!NewGetGroupSizeF->hasFnAttribute(llvm::Attribute::AlwaysInline))
      NewGetGroupSizeF->addFnAttr(llvm::Attribute::AlwaysInline);

    llvm::BasicBlock *BB =
        llvm::BasicBlock::Create(M.getContext(), "", NewGetGroupSizeF);

    auto *ReturnedIntType = llvm::dyn_cast<llvm::IntegerType>(GetGroupSizeF->getReturnType());
    if(!ReturnedIntType)
      return false;

    llvm::Constant *ReturnedValue = llvm::ConstantInt::get(
        M.getContext(), llvm::APInt(ReturnedIntType->getBitWidth(), KnownGroupSize));

    llvm::ReturnInst::Create(M.getContext(), ReturnedValue, BB);

    GetGroupSizeF->replaceNonMetadataUsesWith(NewGetGroupSizeF);
  }

  // Insert __builtin_assume(0 <= local_id); __builtin_assume(local_id < group_size);
  // for every call to GetLocalIdBuiltinName
  if(!insertRangeAssumptionForBuiltinCalls(M, GetLocalIdBuiltinName, 0, KnownGroupSize))
    return false;

  return true;
}

}

KnownGroupSizeOptPass::KnownGroupSizeOptPass(int GroupSizeX, int GroupSizeY, int GroupSizeZ)
    : KnownGroupSizeX{GroupSizeX}, KnownGroupSizeY{GroupSizeY}, KnownGroupSizeZ{GroupSizeZ} {}


llvm::PreservedAnalyses KnownGroupSizeOptPass::run(llvm::Module &M,
                                                        llvm::ModuleAnalysisManager &MAM) {
  if (KnownGroupSizeX > 0) {
    applyKnownGroupSize(M, KnownGroupSizeX, "__hipsycl_sscp_get_local_size_x",
                        "__hipsycl_sscp_get_local_id_x");
  }

  if (KnownGroupSizeY > 0) {
    applyKnownGroupSize(M, KnownGroupSizeY, "__hipsycl_sscp_get_local_size_y",
                        "__hipsycl_sscp_get_local_id_y");
  }

  if (KnownGroupSizeZ > 0) {
    applyKnownGroupSize(M, KnownGroupSizeZ, "__hipsycl_sscp_get_local_size_z",
                        "__hipsycl_sscp_get_local_id_z");
  }

  return llvm::PreservedAnalyses::none();

}


}
}