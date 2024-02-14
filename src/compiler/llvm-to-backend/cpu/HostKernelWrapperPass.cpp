/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2022 Aksel Alpay and contributors
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

#include "hipSYCL/compiler/llvm-to-backend/cpu/HostKernelWrapperPass.hpp"
#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include <algorithm>
#include <iterator>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Argument.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>
#include <llvm/Support/Casting.h>
#include <llvm/Transforms/Utils/Cloning.h>

namespace hipsycl {
namespace compiler {

namespace {
llvm::StoreInst *storeToGlobalVar(llvm::IRBuilderBase Bld, llvm::Value *V,
                                  llvm::StringRef GlobalVarName) {
  auto M = Bld.GetInsertBlock()->getModule();
  auto SizeT = M->getDataLayout().getLargestLegalIntType(M->getContext());
  auto GV = M->getOrInsertGlobal(GlobalVarName, SizeT);
  return Bld.CreateStore(V, GV);
}

void ReplaceUsesOfGVWith(llvm::Function &F, llvm::StringRef GlobalVarName, llvm::Value *To) {
  auto M = F.getParent();
  auto GV = M->getGlobalVariable(GlobalVarName);
  if (!GV)
    return;

  HIPSYCL_DEBUG_INFO << "[SSCP][HostKernelWrapper] RUOGVW: " << *GV << " with " << *To << "\n";
  llvm::SmallVector<llvm::Instruction *> ToErase;
  for (auto U : GV->users()) {
    if (auto I = llvm::dyn_cast<llvm::LoadInst>(U); I && I->getFunction() == &F) {
      HIPSYCL_DEBUG_INFO << "[SSCP][HostKernelWrapper] RUOGVW: " << *I << " with " << *To << "\n";
      I->replaceAllUsesWith(To);
    }
  }
  for (auto I : ToErase)
    I->eraseFromParent();
}

llvm::Function *makeWrapperFunction(llvm::Function &F) {
  auto M = F.getParent();
  auto &Ctx = M->getContext();

  llvm::IRBuilder<> Bld(&F.getEntryBlock());

  auto SizeT = M->getDataLayout().getLargestLegalIntType(Ctx);
  auto Int8T = llvm::IntegerType::get(Ctx, 8);
  auto WorkGroupInfoT =
      llvm::StructType::get(llvm::ArrayType::get(SizeT, 3),       // # groups
                            llvm::ArrayType::get(SizeT, 3),       // group id
                            llvm::ArrayType::get(SizeT, 3),       // local size
                            llvm::PointerType::getUnqual(Int8T)); // local memory ptr
  auto VoidPtrT = llvm::PointerType::getUnqual(Bld.getVoidTy());
  auto UserArgsT = llvm::PointerType::getUnqual(VoidPtrT);

  llvm::SmallVector<llvm::Type *> ArgTypes;
  ArgTypes.push_back(llvm::PointerType::getUnqual(WorkGroupInfoT));
  ArgTypes.push_back(UserArgsT);

  auto WrapperT = llvm::FunctionType::get(Bld.getVoidTy(), ArgTypes, false);
  auto Wrapper = llvm::cast<llvm::Function>(
      M->getOrInsertFunction(("__sscp_wrapper_" + F.getName()).str(), WrapperT, F.getAttributes())
          .getCallee());

  auto WrapperBB = llvm::BasicBlock::Create(Ctx, "entry", Wrapper);
  Bld.SetInsertPoint(WrapperBB);
  auto LoadFromContext = [&](int Array, int D, llvm::StringRef Name) {
    return Bld.CreateLoad(
        SizeT,
        Bld.CreateInBoundsGEP(WorkGroupInfoT, Wrapper->getArg(0),
                              {Bld.getInt64(0), Bld.getInt32(Array), Bld.getInt32(D)}),
        Name);
  };
  std::array<llvm::Value *, 3> NumGroups;
  NumGroups[0] = LoadFromContext(0, 0, "num_groups_x");
  NumGroups[1] = LoadFromContext(0, 1, "num_groups_y");
  NumGroups[2] = LoadFromContext(0, 2, "num_groups_z");

  std::array<llvm::Value *, 3> GroupIds;
  GroupIds[0] = LoadFromContext(1, 0, "group_id_x");
  GroupIds[1] = LoadFromContext(1, 1, "group_id_y");
  GroupIds[2] = LoadFromContext(1, 2, "group_id_z");

  std::array<llvm::Value *, 3> LocalSize;
  LocalSize[0] = LoadFromContext(2, 0, "local_size_x");
  LocalSize[1] = LoadFromContext(2, 1, "local_size_y");
  LocalSize[2] = LoadFromContext(2, 2, "local_size_z");

  auto LocalMemPtr = Bld.CreateLoad(
      llvm::PointerType::get(Ctx, 0),
      Bld.CreateInBoundsGEP(WorkGroupInfoT, Wrapper->getArg(0), {Bld.getInt64(0), Bld.getInt32(3)}),
      "local_mem_ptr");

  llvm::SmallVector<llvm::Value *> Args;

  auto ArgArray = Wrapper->arg_begin() + 1;
  for (int I = 0; I < F.arg_size(); ++I) {
    auto GEP =
        Bld.CreateInBoundsGEP(UserArgsT, ArgArray, llvm::ArrayRef<llvm::Value *>{Bld.getInt32(I)});
    Args.push_back(Bld.CreateLoad(F.getArg(I)->getType(), Bld.CreateLoad(VoidPtrT, GEP)));
  }
  auto FCall = Bld.CreateCall(&F, Args);
  Bld.CreateRetVoid();

  utils::checkedInlineFunction(FCall, "HostKernelWrapperPass");

  for (int I = 0; I < 3; ++I) {
    ReplaceUsesOfGVWith(*Wrapper, NumGroupsGlobalNames[I], NumGroups[I]);
    ReplaceUsesOfGVWith(*Wrapper, GroupIdGlobalNames[I], GroupIds[I]);
    ReplaceUsesOfGVWith(*Wrapper, LocalSizeGlobalNames[I], LocalSize[I]);
  }
  ReplaceUsesOfGVWith(*Wrapper, SscpDynamicLocalMemoryPtrName, LocalMemPtr);

  return Wrapper;
}

} // namespace

llvm::PreservedAnalyses HostKernelWrapperPass::run(llvm::Function &F,
                                                   llvm::FunctionAnalysisManager &AM) {

  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  auto Wrapper = makeWrapperFunction(F);
  F.replaceAllUsesWith(Wrapper);
  std::string Name = F.getName().str();
  F.setName(Name + "_is_wrapped");
  Wrapper->setName(Name);
  HIPSYCL_DEBUG_INFO << "[SSCP][HostKernelWrapper] Created kernel wrapper: " << Wrapper->getName()
                     << "\n";

  F.replaceAllUsesWith(Wrapper);
  // Todo: uncertain if we can erase/remove since we're still tranforming it..? would have to do
  // Module pass? Afaict, it will not be in the final module anyways.. (uncertain why :D)

  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
