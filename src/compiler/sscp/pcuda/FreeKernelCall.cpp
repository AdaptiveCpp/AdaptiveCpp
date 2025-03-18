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

#include <optional>

#include "hipSYCL/compiler/sscp/pcuda/FreeKernelCall.hpp"
#include "hipSYCL/compiler/sscp/KernelOutliningPass.hpp"
#include "hipSYCL/compiler/cbs/IRUtils.hpp"
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassManager.h>

namespace hipsycl {
namespace compiler {

namespace {
static const char* IsDeviceGVName = "__acpp_sscp_is_device";
static const char* HcfObjectIdGVName = "__acpp_local_sscp_hcf_object_id";
static const char* KernelCallFunctionName = "__pcudaKernelCall";

llvm::Type* getVoidPtrType(llvm::Module* M, unsigned AS = 0) {
#if LLVM_VERSION_MAJOR >= 16
    return llvm::PointerType::get(M->getContext(), AS);
#else
    return llvm::PointerType::get(llvm::Type::getInt8Ty(M->getContext()), AS);
#endif

}

llvm::GlobalVariable* getSSCPIsDevicePassGV(llvm::Module* M) {
  if(auto* GV = M->getGlobalVariable(IsDeviceGVName))
    return GV;
  else {
    llvm::GlobalVariable *NewGV =
        new llvm::GlobalVariable{*M,      llvm::Type::getInt32Ty(M->getContext()),
                                 false,   llvm::GlobalVariable::ExternalLinkage,
                                 nullptr, IsDeviceGVName};
    NewGV->setAlignment(M->getDataLayout().getABIIntegerTypeAlignment(32));
    return NewGV;
  }
}

llvm::GlobalVariable* getHcfObjectIdGV(llvm::Module* M) {
  if(auto* GV = M->getGlobalVariable(HcfObjectIdGVName))
    return GV;
  else {
    // Needs to be non-const, externally-linked until target-separation
    llvm::GlobalVariable *NewGV =
        new llvm::GlobalVariable{*M,      llvm::Type::getInt64Ty(M->getContext()),
                                 false,   llvm::GlobalVariable::ExternalLinkage,
                                 nullptr, HcfObjectIdGVName};
    NewGV->setAlignment(M->getDataLayout().getABIIntegerTypeAlignment(64));
    return NewGV;
  }
}

llvm::GlobalVariable* generateKernelName(llvm::Function* F) {
  std::string Name = F->getName().str();
  std::string GVName = "__acpp_free_kernel_name_" + Name;

  if (auto *GV = F->getParent()->getGlobalVariable(GVName)) {
    return GV;
  } else {
    llvm::Constant *Initializer = llvm::ConstantDataArray::getRaw(
        Name + '\0', Name.size() + 1, llvm::Type::getInt8Ty(F->getContext()));

    llvm::Module* M = F->getParent();
    llvm::GlobalVariable *NewGV = new llvm::GlobalVariable{
        *M, Initializer->getType(), true, llvm::GlobalValue::InternalLinkage, Initializer, GVName};
    return NewGV;
  }
}

llvm::GlobalVariable* getKernelSpecificStorage(llvm::Module* M, llvm::Function* F) {
  std::string Name = F->getName().str();
  std::string GVName = "__acpp_kernel_specific_storage_" + Name;

  if(auto* GV = M->getGlobalVariable(GVName))
    return GV;
  else {
    llvm::Type* VoidPtrType = getVoidPtrType(M);

    auto *NullInitializer =
        llvm::ConstantPointerNull::get(llvm::dyn_cast<llvm::PointerType>(VoidPtrType));
    llvm::GlobalVariable *NewGV =
        new llvm::GlobalVariable{*M,      VoidPtrType,
                                 false,   llvm::GlobalVariable::InternalLinkage,
                                 NullInitializer, GVName};
    NewGV->setAlignment(M->getDataLayout().getABITypeAlign(VoidPtrType));
    return NewGV;
  }
}

llvm::Function* getKernelLaunchFunction(llvm::Module* M) {
  if(auto* F = M->getFunction(KernelCallFunctionName)) {
    return F;
  } else {
    llvm::SmallVector<llvm::Type*> ParamTs;
    // const char*
    // With typed ptrs, void* is mapped to i8*, so void* is equivalent
    // to i8* type when we have typed pointers.
    ParamTs.push_back(getVoidPtrType(M));

    // void**
#if LLVM_VERSION_MAJOR >= 16
    auto VoidVoidPtrType = llvm::PointerType::get(M->getContext(), 0);
#else
    auto* VoidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(M->getContext()), 0);
    auto* VoidVoidPtrType = llvm::PointerType::get(VoidPtrType, 0);
#endif
    ParamTs.push_back(VoidVoidPtrType);

    // std::size_t
    ParamTs.push_back(llvm::Type::getInt64Ty(M->getContext()));
    // void**
    ParamTs.push_back(VoidVoidPtrType);

    auto FC = M->getOrInsertFunction(KernelCallFunctionName,
                           llvm::FunctionType::get(llvm::Type::getInt32Ty(M->getContext()),
                                                   llvm::ArrayRef<llvm::Type *>{ParamTs}, false));
    llvm::Function* NewDeclaration = llvm::dyn_cast<llvm::Function>(FC.getCallee());
    NewDeclaration->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
    return NewDeclaration;
  }
}

// Generate a function with the following definition:
//
// void kernel(args) { if(__acpp_sscp_is_device) original_kernel(args) else { /* Perform host kernel
// call */ } }
llvm::Function* generateKernelWrapper(llvm::Function* KernelF) {
  llvm::Module* M = KernelF->getParent();
  auto IsArgByValue =
      KernelArgumentCanonicalizationPass::areFreeKernelFunctionParamsByValue(KernelF);

  if(!KernelF->getReturnType()->isVoidTy()) {
    HIPSYCL_DEBUG_ERROR << "FreeKernelCallPass: Kernel " << KernelF->getName().str()
                        << " does not have void return type\n";
    return nullptr;
  }

  std::string Name = KernelF->getName().str();
  KernelF->setName(Name+".original");

  llvm::Function *NewF = llvm::dyn_cast<llvm::Function>(
      KernelF->getParent()->getOrInsertFunction(Name, KernelF->getFunctionType()).getCallee());
  
  for(auto Attr: KernelF->getAttributes().getFnAttrs())
    NewF->addFnAttr(Attr);
  for(int i = 0; i < KernelF->getFunctionType()->getNumParams(); ++i)
    for(auto Attr : KernelF->getAttributes().getParamAttrs(i))
      NewF->addParamAttr(i, Attr);
  
  NewF->setLinkage(KernelF->getLinkage());
  NewF->setCallingConv(KernelF->getCallingConv());
  
  llvm::GlobalVariable* IsDevicePassGV = getSSCPIsDevicePassGV(NewF->getParent());

  llvm::BasicBlock *EntryBlock = llvm::BasicBlock::Create(NewF->getContext(), "", NewF);

  llvm::LoadInst *LoadSSCPIsDevice = new llvm::LoadInst(llvm::Type::getInt32Ty(NewF->getContext()),
                                                        IsDevicePassGV, "", EntryBlock);
  auto *Icmp = llvm::ICmpInst::Create(
      llvm::Instruction::OtherOps::ICmp, llvm::ICmpInst::Predicate::ICMP_NE, LoadSSCPIsDevice,
      llvm::ConstantInt::get(LoadSSCPIsDevice->getType(), llvm::APInt{32, 0}), "", EntryBlock);

  
  llvm::BasicBlock* IfBranch = llvm::BasicBlock::Create(NewF->getContext(), "", NewF);
  llvm::BasicBlock* ElseBranch = llvm::BasicBlock::Create(NewF->getContext(), "", NewF);
  llvm::BasicBlock* EndBlock = llvm::BasicBlock::Create(NewF->getContext(), "", NewF);

  llvm::BranchInst::Create(IfBranch, ElseBranch, Icmp, EntryBlock);

  // If Branch
  {
    llvm::SmallVector<llvm::Value*, 16> Args;
    for(int i = 0; i < NewF->getFunctionType()->getNumParams(); ++i)
      Args.push_back(NewF->getArg(i));
    llvm::CallInst::Create(llvm::FunctionCallee(KernelF), llvm::ArrayRef<llvm::Value *>{Args}, "",
                           IfBranch);
    llvm::BranchInst::Create(EndBlock, IfBranch);
  }
  // Else Branch
  {
 
    auto AllocaAS = M->getDataLayout().getAllocaAddrSpace();
#if LLVM_VERSION_MAJOR >= 16
    auto* VoidPtrType = llvm::PointerType::get(NewF->getContext(), AllocaAS);
    auto* VoidVoidPtrType = llvm::PointerType::get(NewF->getContext(), AllocaAS);
#else
    auto* VoidPtrType = llvm::PointerType::get(llvm::Type::getInt8Ty(NewF->getContext()), AllocaAS);
    auto* VoidVoidPtrType = llvm::PointerType::get(VoidPtrType, AllocaAS);
#endif

    auto *PtrArrayTy = llvm::ArrayType::get(VoidPtrType, NewF->getFunctionType()->getNumParams());
    llvm::AllocaInst *ParamPtrList = new llvm::AllocaInst(PtrArrayTy, AllocaAS, "", ElseBranch);
    for(int i = 0; i < NewF->getFunctionType()->getNumParams(); ++i) {

      llvm::Value *ParamValue = nullptr;
      if(NewF->getFunctionType()->getParamType(i)->isPointerTy() && IsArgByValue[i]) {
        // A pointer could be either an actual pointer, or a pointer to a ByVal struct,
        // in which case we need to pass in the pointer directly, not an alloca pointer holding the
        // pointer
        ParamValue = NewF->getArg(i);
      } else {
        ParamValue = new llvm::AllocaInst{NewF->getFunctionType()->getParamType(i),
                               AllocaAS, "", ElseBranch};
        auto* SI = new llvm::StoreInst(NewF->getArg(i), ParamValue, ElseBranch);
      }
      // In case we still have typed ptrs
      auto *VoidPtrParam = new llvm::BitCastInst(ParamValue, VoidPtrType, "", ElseBranch);
      // Store VoidPtrParam in ParamPtrList
      llvm::SmallVector<llvm::Value*> Indices;
      Indices.push_back(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), llvm::APInt{32, 0}));
      Indices.push_back(
          llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), llvm::APInt{32, static_cast<uint64_t>(i)}));

      auto *GEPInst = llvm::GetElementPtrInst::CreateInBounds(
          PtrArrayTy, ParamPtrList, llvm::ArrayRef<llvm::Value *>{Indices}, "", ElseBranch);
      
      new llvm::StoreInst(VoidPtrParam, GEPInst, ElseBranch);
    }

    auto *VoidVoidPtrParamList = new llvm::BitCastInst(ParamPtrList, VoidVoidPtrType, "", ElseBranch);
    llvm::LoadInst *HcfObjectId = new llvm::LoadInst(llvm::Type::getInt64Ty(M->getContext()),
                                                     getHcfObjectIdGV(M), "", ElseBranch);

    
    auto* KernelNameGV = generateKernelName(NewF);
    llvm::SmallVector<llvm::Value*> ZeroZeroIndices{
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), llvm::APInt{32, 0}),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M->getContext()), llvm::APInt{32, 0})
    };
    auto *KernelNamePtr = llvm::GetElementPtrInst::CreateInBounds(
        KernelNameGV->getValueType(), KernelNameGV, llvm::ArrayRef<llvm::Value *>{ZeroZeroIndices},
        "", ElseBranch);

    llvm::Function* KLF = getKernelLaunchFunction(M);
    llvm::GlobalVariable* KernelSpecificStorage = getKernelSpecificStorage(M, NewF);

    llvm::SmallVector<llvm::Value*> KLFArgs {
      KernelNamePtr,
      VoidVoidPtrParamList,
      HcfObjectId,
      KernelSpecificStorage
    };

    llvm::CallInst::Create(llvm::FunctionCallee{KLF->getFunctionType(), KLF},
                           llvm::ArrayRef<llvm::Value *>{KLFArgs}, "", ElseBranch);

    llvm::BranchInst::Create(EndBlock, ElseBranch);
  }

  llvm::ReturnInst::Create(NewF->getContext(), EndBlock);

  KernelF->replaceUsesWithIf(NewF, [&](llvm::Use &U) {
    if (auto *CB = llvm::dyn_cast<llvm::CallBase>(U.getUser())) {
      if (CB->getCalledFunction() == KernelF && CB->getParent()->getParent() == NewF)
        return false;
    }
    return true;
  });

  return NewF;
}

}

llvm::PreservedAnalyses FreeKernelCallPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  llvm::SmallPtrSet<llvm::Function *, 16> FreeKernels;

  utils::findFunctionsWithStringAnnotations(M, [&](llvm::Function *F, llvm::StringRef Annotation) {
    if (Annotation.compare("acpp_free_kernel") == 0) {
      FreeKernels.insert(F);
    }
  });

  for(auto* F : FreeKernels) {
    generateKernelWrapper(F);
  }

  return llvm::PreservedAnalyses::none();
}
}
}

