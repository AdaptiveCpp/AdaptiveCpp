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

#include "hipSYCL/compiler/llvm-to-backend/host/HostKernelWrapperPass.hpp"

#include "hipSYCL/common/debug.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include "hipSYCL/compiler/cbs/SplitterAnnotationAnalysis.hpp"
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

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
#include <llvm/IR/Metadata.h>
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

void replaceUsesOfGVWith(llvm::Function &F, llvm::StringRef GlobalVarName, llvm::Value *To) {
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

/*
 * This creates a wrapper function for a kernel function that takes the following arguments:
 * - A pointer to a struct containing {num_groups, group_id, local_size, local_mem_ptr}
 * - A pointer to pointers (the actual kernel arguments)
 * The wrapper function will extract the work group information and the user arguments from the
 * provided pointers and call the kernel function with the extracted arguments.
 * The original kernel is then inlined and all calls to the original kernel are replaced with the
 * wrapper.
 * This makes calling the kernel from the host code straighforward, as only the work group info
 * struct and the user arguments need to be passed to the wrapper.
 */
llvm::Function *makeWrapperFunction(llvm::Function &F, std::int64_t DynamicLocalMemSize) {
  auto M = F.getParent();
  auto &Ctx = M->getContext();

  llvm::IRBuilder<> Bld(&F.getEntryBlock());

  auto SizeT = M->getDataLayout().getLargestLegalIntType(Ctx);
  auto WorkGroupInfoT =
      llvm::StructType::get(llvm::ArrayType::get(SizeT, 3),                 // # groups
                            llvm::ArrayType::get(SizeT, 3),                 // group id
                            llvm::ArrayType::get(SizeT, 3),                 // local size
                            llvm::PointerType::getUnqual(Bld.getInt8Ty())); // local memory size
  auto VoidPtrT = llvm::PointerType::getUnqual(Bld.getInt8Ty());
  auto UserArgsT = llvm::PointerType::getUnqual(VoidPtrT);

  llvm::SmallVector<llvm::Type *> ArgTypes;
  ArgTypes.push_back(llvm::PointerType::getUnqual(WorkGroupInfoT));
  ArgTypes.push_back(UserArgsT);

  std::string FName = F.getName().str();
  F.setName(FName + "_original");

  auto WrapperT = llvm::FunctionType::get(Bld.getVoidTy(), ArgTypes, false);
  auto Wrapper = llvm::cast<llvm::Function>(
      M->getOrInsertFunction(FName, WrapperT, F.getAttributes()).getCallee());
  Wrapper->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);

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
      VoidPtrT,
      Bld.CreateInBoundsGEP(WorkGroupInfoT, Wrapper->getArg(0), {Bld.getInt64(0), Bld.getInt32(3)}),
      "local_mem_ptr");

  if (DynamicLocalMemSize >= 0)
    LocalMemPtr->setMetadata(
        llvm::LLVMContext::MD_dereferenceable,
        llvm::MDNode::get(Ctx, {llvm::ConstantAsMetadata::get(Bld.getInt64(DynamicLocalMemSize))}));

  llvm::SmallVector<llvm::Value *> Args;

  auto ArgArray = Wrapper->arg_begin() + 1;
  for (int I = 0; I < F.arg_size(); ++I) {

    if IS_OPAQUE(ArgArray->getType()) {
      auto GEP = Bld.CreateInBoundsGEP(UserArgsT, ArgArray,
                                       llvm::ArrayRef<llvm::Value *>{Bld.getInt32(I)});
      Args.push_back(Bld.CreateLoad(F.getArg(I)->getType(), Bld.CreateLoad(VoidPtrT, GEP)));
    } else {
#if HAS_TYPED_PTR // otherwise, IS_OPAQUE is always true
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
      auto GEP = Bld.CreateInBoundsGEP(UserArgsT->getNonOpaquePointerElementType(), ArgArray,
                                       llvm::ArrayRef<llvm::Value *>{Bld.getInt32(I)});
#pragma GCC diagnostic pop
      auto CastedPtr = Bld.CreatePointerCast(Bld.CreateLoad(VoidPtrT, GEP),
                                             llvm::PointerType::getUnqual(F.getArg(I)->getType()));
      Args.push_back(Bld.CreateLoad(F.getArg(I)->getType(), CastedPtr));
#endif
    }
  }
  auto FCall = Bld.CreateCall(&F, Args);
  Bld.CreateRetVoid();

  utils::checkedInlineFunction(FCall, "HostKernelWrapperPass");

  for (int I = 0; I < 3; ++I) {
    replaceUsesOfGVWith(*Wrapper, cbs::NumGroupsGlobalNames[I], NumGroups[I]);
    replaceUsesOfGVWith(*Wrapper, cbs::GroupIdGlobalNames[I], GroupIds[I]);
    replaceUsesOfGVWith(*Wrapper, cbs::LocalSizeGlobalNames[I], LocalSize[I]);
  }
  replaceUsesOfGVWith(*Wrapper, cbs::SscpDynamicLocalMemoryPtrName, LocalMemPtr);

  F.setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
  F.replaceAllUsesWith(Wrapper);
  // can't erase here, since the original function is still transformed.

  return Wrapper;
}

} // namespace

llvm::PreservedAnalyses HostKernelWrapperPass::run(llvm::Function &F,
                                                   llvm::FunctionAnalysisManager &AM) {

  auto &MAM = AM.getResult<llvm::ModuleAnalysisManagerFunctionProxy>(F);
  auto *SAA = MAM.getCachedResult<SplitterAnnotationAnalysis>(*F.getParent());
  if (!SAA || !SAA->isKernelFunc(&F))
    return llvm::PreservedAnalyses::all();

  auto Wrapper = makeWrapperFunction(F, DynamicLocalMemSize);

  HIPSYCL_DEBUG_INFO << "[SSCP][HostKernelWrapper] Created kernel wrapper: " << Wrapper->getName()
                     << "\n";

  return llvm::PreservedAnalyses::none();
}

} // namespace compiler
} // namespace hipsycl
