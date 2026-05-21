/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationPass.hpp"
#include "hipSYCL/compiler/llvm-to-backend/metal/PointerTranslationAnnotationPass.hpp"

#include "hipSYCL/common/debug.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Type.h>

using namespace llvm;

namespace hipsycl {
namespace compiler {

namespace {

static constexpr const char *SymbolIdFuncName = "__acpp_sscp_metal_symbol_id";
static constexpr const char *AddrDiffSymbolName = "__acpp_sscp_metal_gpu_to_host_addr_diff";

Function *getOrCreateSymbolIdDecl(Module &M) {
  if (Function *F = M.getFunction(SymbolIdFuncName)) {
    return F;
  }

  LLVMContext &Ctx = M.getContext();
  FunctionType *FT = FunctionType::get(Type::getInt64Ty(Ctx), {PointerType::getUnqual(Ctx)}, /*isVarArg=*/false);
  return Function::Create(FT, GlobalValue::ExternalLinkage, SymbolIdFuncName, M);
}

GlobalVariable *getOrCreateAddrDiffStrGlobal(Module &M) {
  if (GlobalVariable *GV = M.getNamedGlobal(AddrDiffSymbolName)) {
    return GV;
  }

  LLVMContext &Ctx = M.getContext();
  Constant *Str = ConstantDataArray::getString(Ctx, AddrDiffSymbolName, /*AddNull=*/true);
  auto *GV = new GlobalVariable(M, Str->getType(), /*isConstant=*/true, GlobalValue::PrivateLinkage, Str, AddrDiffSymbolName);
  GV->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);
  return GV;
}

CallInst *getOrInsertDeltaInEntryBlock(Function &F, Function *SymbolIdFn, GlobalVariable *AddrDiffGV) {
  BasicBlock &Entry = F.getEntryBlock();
  for (Instruction &I : Entry) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (CI->getCalledFunction() == SymbolIdFn && CI->getArgOperand(0) == AddrDiffGV) {
        return CI;
      }
    }
  }

  IRBuilder<> B(&*Entry.getFirstInsertionPt());
  return B.CreateCall(SymbolIdFn, {AddrDiffGV}, "gpu_host_delta");
}

} // anonymous namespace

PointerTranslationPass::PointerTranslationPass(unsigned GlobalAS)
  : GlobalAS(GlobalAS)
{}

PreservedAnalyses PointerTranslationPass::run(Module &M, ModuleAnalysisManager &MAM) {
  Function *SymbolIdFn = nullptr;
  GlobalVariable *AddrDiffGV = nullptr;

  unsigned LoadsTranslated = 0;
  unsigned StoresTranslated = 0;

  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    SmallVector<LoadInst *, 16> AnnotatedLoads;
    SmallVector<StoreInst *, 16> AnnotatedStores;

    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (auto *LI = dyn_cast<LoadInst>(&I)) {
          if (LI->getMetadata(MetalPtrLoadNeedsTranslationMD)) {
            AnnotatedLoads.push_back(LI);
          }
        } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
          if (SI->getMetadata(MetalPtrStoreNeedsTranslationMD)) {
            AnnotatedStores.push_back(SI);
          }
        }
      }
    }

    if (AnnotatedLoads.empty() && AnnotatedStores.empty()) {
      continue;
    }

    if (!SymbolIdFn) {
      SymbolIdFn = getOrCreateSymbolIdDecl(M);
    }
    if (!AddrDiffGV) {
      AddrDiffGV = getOrCreateAddrDiffStrGlobal(M);
    }
    CallInst *Delta = getOrInsertDeltaInEntryBlock(F, SymbolIdFn, AddrDiffGV);

    // Values in memory use canonical host-side addresses; kernel registers use
    // GPU addresses. Nulls are the only values not shifted by the linear map.
    for (LoadInst *LI : AnnotatedLoads) {
      IRBuilder<> B(LI->getNextNode());
      auto *PtrTy = cast<PointerType>(LI->getType());
      Value *NullPtr = ConstantPointerNull::get(PtrTy);
      Value *IsNull = B.CreateICmpEQ(LI, NullPtr, "ptr_loaded_is_null");
      Value *Gep = B.CreateGEP(Type::getInt8Ty(M.getContext()), LI, Delta, "ptr_gpu_raw");
      Value *Adjusted = B.CreateSelect(IsNull, LI, Gep, "ptr_gpu_adjusted");
      LI->replaceUsesWithIf(Adjusted, [&](Use &U) {
        return U.getUser() != IsNull && U.getUser() != Gep && U.getUser() != Adjusted;
      });
      LI->setMetadata(MetalPtrLoadNeedsTranslationMD, nullptr);
      ++LoadsTranslated;
      HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated load in "
                         << F.getName() << "\n";
    }

    for (StoreInst *SI : AnnotatedStores) {
      IRBuilder<> B(SI);
      Value *Val = SI->getValueOperand();
      auto *PtrTy = cast<PointerType>(Val->getType());
      Value *NullPtr = ConstantPointerNull::get(PtrTy);
      Value *IsNull = B.CreateICmpEQ(Val, NullPtr, "ptr_stored_is_null");
      Value *NegDelta = B.CreateNeg(Delta, "neg_gpu_host_delta");
      Value *Gep = B.CreateGEP(Type::getInt8Ty(M.getContext()), Val, NegDelta, "ptr_host_raw");
      Value *Adjusted = B.CreateSelect(IsNull, Val, Gep, "ptr_host_adjusted");
      SI->setOperand(0, Adjusted);
      SI->setMetadata(MetalPtrStoreNeedsTranslationMD, nullptr);
      ++StoresTranslated;
      HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated store in "
                         << F.getName() << "\n";
    }
  }

  HIPSYCL_DEBUG_INFO << "Metal PointerTranslationPass: translated "
                     << LoadsTranslated << " load(s) and "
                     << StoresTranslated << " store(s).\n";

  return (LoadsTranslated + StoresTranslated) > 0
    ? PreservedAnalyses::none()
    : PreservedAnalyses::all();
}

} // namespace compiler
} // namespace hipsycl
