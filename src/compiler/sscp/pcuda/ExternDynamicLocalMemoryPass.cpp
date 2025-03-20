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

#include "hipSYCL/compiler/sscp/pcuda/ExternDynamicLocalMemoryPass.hpp"
#include "hipSYCL/compiler/utils/ConstantExpressions.hpp"
#include "hipSYCL/compiler/utils/LLVMUtils.hpp"

#include <llvm/ADT/DenseMap.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Instructions.h>

namespace hipsycl {
namespace compiler {

namespace {

static const char *SscpDynamicLocalMemoryBuiltinIdentifier = "__acpp_sscp_get_dynamic_local_memory";

llvm::Function* getSscpDynamicLocalMemoryBuiltin(llvm::Module& M, unsigned LocalAS) {
  if(auto* F = M.getFunction(SscpDynamicLocalMemoryBuiltinIdentifier))
    return F;
  else {
#if LLVM_VERSION_MAJOR >= 16
    auto *LocalVoidPtrType = llvm::PointerType::get(M.getContext(), LocalAS);
#else
    auto *LocalVoidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(M.getContext()), LocalAS);
#endif
    llvm::Function *BuiltinDeclaration = llvm::dyn_cast<llvm::Function>(
        M.getOrInsertFunction(SscpDynamicLocalMemoryBuiltinIdentifier, LocalVoidPtrType)
            .getCallee());
    assert(BuiltinDeclaration);
    return BuiltinDeclaration;
  }
}

static const char* LocalMemoryABITag = "__acpp_local_memory_tag__";

void replaceGVsWithDynamicLocalMemory(llvm::Module &M,
                                      const llvm::SmallVector<llvm::GlobalVariable *> &GVs,
                                      unsigned LocalMemAS) {

  llvm::Function *DynamicLocalMemBuiltin = getSscpDynamicLocalMemoryBuiltin(M, LocalMemAS);

  for (auto *GV : GVs) {
    for(auto* U : GV->users()) {
      if(auto* CE = llvm::dyn_cast<llvm::ConstantExpr>(U))
        utils::unfoldConstantExpression(CE);
    }

    llvm::SmallDenseMap<llvm::Function *, llvm::Value *> FunctionToDynamicLocalMemMap;

    for (auto &U : GV->uses()) {
      if (auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
        llvm::Function *F = I->getParent()->getParent();
        if (FunctionToDynamicLocalMemMap.find(F) == FunctionToDynamicLocalMemMap.end()) {
          if (!F->isDeclaration()) {
            auto InsertionPt = llvmutils::makeInsertionPoint(&(*F->getEntryBlock().getFirstInsertionPt()));
            llvm::Instruction *BuiltinCall = llvm::CallInst::Create(
                llvm::FunctionCallee{DynamicLocalMemBuiltin}, "", InsertionPt);
#if LLVM_VERSION_MAJOR < 17
            llvm::Type *CastAsType = llvm::PointerType::getWithSamePointeeType(
                llvm::dyn_cast<llvm::PointerType>(DynamicLocalMemBuiltin->getReturnType()),
                GV->getAddressSpace());
#else
            llvm::Type *CastAsType = llvm::PointerType::get(M.getContext(), GV->getAddressSpace());
#endif
            llvm::Instruction *ASCast =
                new llvm::AddrSpaceCastInst{BuiltinCall, CastAsType, "", InsertionPt};

            llvm::Instruction *BitCast =
                new llvm::BitCastInst(ASCast, GV->getType(), "", InsertionPt);

            FunctionToDynamicLocalMemMap[F] = BitCast;
          }
        }
      }
    }

    for (auto Entry : FunctionToDynamicLocalMemMap) {
      llvm::Function *TargetF = Entry.first;
      llvm::Value *ReplacementI = Entry.second;

      GV->replaceUsesWithIf(ReplacementI, [&](llvm::Use &U) {
        if (auto *I = llvm::dyn_cast<llvm::Instruction>(U.getUser())) {
          if (I->getParent() && I->getParent()->getParent()) {
            auto *OwningF = I->getParent()->getParent();
            if (OwningF == TargetF) {
              return true;
            }
          }
        }
        return false;
      });
    }
  }
}

} // namespace

ExternDynamicLocalMemoryPass::ExternDynamicLocalMemoryPass(unsigned LocalMemAddressSpace,
                                                           bool IsDevicePass)
    : LocalMemAS{LocalMemAddressSpace}, IsDevice{IsDevicePass} {}

llvm::PreservedAnalyses ExternDynamicLocalMemoryPass::run(llvm::Module& M, llvm::ModuleAnalysisManager& MAM) {
  llvm::SmallVector<llvm::GlobalVariable*> GVs;

  for (auto &GV : M.globals()) {
    if (GV.getLinkage() == llvm::GlobalValue::ExternalLinkage &&
        GV.getName().contains(LocalMemoryABITag)) {
      GVs.push_back(&GV);
    }
  }

  if(IsDevice) {
   replaceGVsWithDynamicLocalMemory(M, GVs, LocalMemAS);
  } else {
    for(auto& GV : GVs) {
      GV->setLinkage(llvm::GlobalValue::InternalLinkage);
      GV->setInitializer(llvm::UndefValue::get(GV->getValueType()));
    }
  }

  return llvm::PreservedAnalyses::none();
}

}
}
