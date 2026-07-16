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

#include "hipSYCL/compiler/sscp/HcfRegistrationPass.hpp"


#include <llvm/ADT/SmallVector.h>

#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Type.h>
#include <llvm/Pass.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>



namespace hipsycl::compiler {
namespace {


// Helper function to add to llvm.compiler.used
void appendToCompilerUsed(llvm::Module &M, llvm::GlobalVariable *GV) {
  llvm::LLVMContext &Ctx = M.getContext();
  llvm::Type *I8PtrTy = llvm::PointerType::get(Ctx, 0);

  // Get the existing llvm.compiler.used array, or create one
  llvm::GlobalVariable *UsedArray = M.getGlobalVariable("llvm.compiler.used");
  llvm::SmallVector<llvm::Constant *, 8> UsedElements;

  if (UsedArray) {
    // Extract existing elements
    llvm::ConstantArray *InitArray = llvm::cast<llvm::ConstantArray>(UsedArray->getInitializer());
    for (llvm::Value *V : InitArray->operands()) {
      UsedElements.push_back(llvm::cast<llvm::Constant>(V));
    }
    UsedArray->eraseFromParent();
  }

  // Add our new global (cast to i8* as required by llvm.used)
  UsedElements.push_back(llvm::ConstantExpr::getPointerCast(GV, I8PtrTy));

  // Create the new array and global variable
  llvm::ArrayType *ATy = llvm::ArrayType::get(I8PtrTy, UsedElements.size());
  llvm::Constant *NewInit = llvm::ConstantArray::get(ATy, UsedElements);

  llvm::GlobalVariable *NewUsedArray = new llvm::GlobalVariable(
      M, ATy, false, llvm::GlobalValue::AppendingLinkage, NewInit, "llvm.compiler.used");
  NewUsedArray->setSection("llvm.metadata");
}

void makeFunctionPtrDiscoverablePerSection(llvm::Module &M, llvm::Function *DeviceInitFunc) {
  llvm::LLVMContext &Ctx = M.getContext();

  static const char *SectionName = ".__acpp_sscp_hcf_registration_func_ptrs";

  llvm::Type *FuncPtrType = llvm::PointerType::get(Ctx, 0);

  llvm::GlobalVariable *RegVar = new llvm::GlobalVariable(
      M, FuncPtrType,
      true,
      llvm::GlobalValue::InternalLinkage,
      llvm::ConstantExpr::getPointerCast(DeviceInitFunc, FuncPtrType),
      "__acpp_sscp_invoke_register_hcf_addr"
  );

  RegVar->setSection(SectionName);
  
  // Prevent DCE: We must tell LLVM that this global is "used" even if no function calls it.
  appendToCompilerUsed(M, RegVar);
}

// Helper function to modify @llvm.global_ctors or @llvm.global_dtors
static void modifyGlobalList(llvm::Module &M, const std::string &ListName, llvm::Function *F, int Priority,
                             bool Prepend) {
  if (!F)
    return;

  auto &C = M.getContext();
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(C);
  llvm::PointerType *FuncPtrTy = llvm::PointerType::getUnqual(F->getFunctionType());
  llvm::PointerType *Int8PtrTy = llvm::PointerType::getUnqual(llvm::Type::getInt8Ty(C));

  // The standard LLVM struct type for ctors/dtors: { i32, ptr, ptr }
  // If the global variable already exists, we extract its exact struct type
  // to ensure layout compatibility, otherwise we create the standard one.
  llvm::StructType *StructTy = llvm::StructType::get(Int32Ty, FuncPtrTy, Int8PtrTy);

  llvm::SmallVector<llvm::Constant*> ExistingElements;

  // Check if the global list already exists
  llvm::GlobalVariable *GV = M.getGlobalVariable(ListName);
  if (GV) {
    // Extract the existing struct type from the array to be 100% safe
    llvm::ArrayType *ArrTy = llvm::dyn_cast<llvm::ArrayType>(GV->getValueType());
    if (ArrTy) {
      StructTy = llvm::dyn_cast<llvm::StructType>(ArrTy->getElementType());
      llvm::ConstantArray *InitList = llvm::dyn_cast<llvm::ConstantArray>(GV->getInitializer());
      if (InitList && StructTy) {
        for (llvm::Value *Op : InitList->operands()) {
          ExistingElements.push_back(llvm::cast<llvm::Constant>(Op));
        }
      }
    }
    // Remove the old global variable; we will recreate it
    GV->eraseFromParent();
  }

  // Create the new entry: { Priority, FunctionPtr, Null }
  llvm::Constant *NullPtr = llvm::ConstantPointerNull::get(Int8PtrTy);
  llvm::Constant *NewEntry =
      llvm::ConstantStruct::get(StructTy, llvm::ConstantInt::get(Int32Ty, Priority), F, NullPtr);

  // Insert at the beginning or append to the end
  if (Prepend) {
    ExistingElements.insert(ExistingElements.begin(), NewEntry);
  } else {
    ExistingElements.push_back(NewEntry);
  }

  // Create the new array initializer
  llvm::ArrayType *NewArrTy = llvm::ArrayType::get(StructTy, ExistingElements.size());
  llvm::Constant *NewInit = llvm::ConstantArray::get(NewArrTy, ExistingElements);

  // Create the new Global Variable.
  // It MUST use AppendingLinkage for @llvm.global_ctors/dtors
  GV = new llvm::GlobalVariable(M, NewArrTy, true, llvm::GlobalValue::AppendingLinkage, NewInit,
                                ListName);
}

// These two functions return function declarations defined by the runtime:
// extern "C" void __acpp_register_hcf(const char* hcf, unsigned long long size);
// extern "C" void __acpp_unregister_hcf(unsigned long long hcf_object_id);

static constexpr const char RegisterHcfName [] = "__acpp_register_hcf";
static constexpr const char UnregisterHcfName [] = "__acpp_unregister_hcf";

llvm::Function *getHcfRegistrationFunction(llvm::Module *M) {
  if (auto *F = M->getFunction(RegisterHcfName))
    return F;

  llvm::Type *Params[] = {llvm::PointerType::get(M->getContext(), 0),
                          llvm::Type::getInt64Ty(M->getContext())};
  llvm::FunctionType *FuncType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), Params, false);
  return llvm::Function::Create(FuncType, llvm::GlobalValue::ExternalLinkage, RegisterHcfName, *M);
}


llvm::Function *getHcfUnregistrationFunction(llvm::Module *M) {
  if (auto *F = M->getFunction(UnregisterHcfName))
    return F;

  llvm::Type *Params[] = {llvm::Type::getInt64Ty(M->getContext())};
  llvm::FunctionType *FuncType =
      llvm::FunctionType::get(llvm::Type::getVoidTy(M->getContext()), Params, false);
  return llvm::Function::Create(FuncType, llvm::GlobalValue::ExternalLinkage, UnregisterHcfName, *M);
}

static constexpr const char InvokeRegisterHcfName [] = "__acpp_invoke_register_hcf";
static constexpr const char InvokeUnregisterHcfName [] = "__acpp_invoke_unregister_hcf";

static constexpr const char HcfContentGVName [] = "__acpp_local_sscp_hcf_content.initialized";
static constexpr const char HcfIdGVName [] = "__acpp_local_sscp_hcf_object_id.initialized";
static constexpr const char HcfSizeGVName [] = "__acpp_local_sscp_hcf_object_size.initialized";

bool generateCallerFunctions(llvm::Module &M) {
  llvm::IRBuilder<> Builder(M.getContext());

  // Get the previously declared functions
  llvm::Function *registerFunc = getHcfRegistrationFunction(&M);
  llvm::Function *unregisterFunc = getHcfUnregistrationFunction(&M);

  if (!registerFunc || !unregisterFunc) {
    return false;
  }

  llvm::Type *voidTy = llvm::Type::getVoidTy(M.getContext());
  llvm::Type *ptrTy = llvm::PointerType::get(M.getContext(), 0);
  llvm::Type *i64Ty = llvm::Type::getInt64Ty(M.getContext());

  llvm::GlobalVariable *hcfContent = M.getGlobalVariable(HcfContentGVName, true);
  llvm::GlobalVariable *hcfId = M.getGlobalVariable(HcfIdGVName, true);
  llvm::GlobalVariable *hcfSize = M.getGlobalVariable(HcfSizeGVName, true);

  if(!hcfContent || !hcfId || !hcfSize) {
    return false;
  }
  

  // Generate void @__acpp_invoke_register_hcf()
  llvm::FunctionType *callRegFuncType = llvm::FunctionType::get(voidTy, false);
  llvm::Function *callRegFunc = llvm::Function::Create(
      callRegFuncType, llvm::GlobalValue::InternalLinkage, InvokeRegisterHcfName, &M);

  llvm::BasicBlock *regEntry = llvm::BasicBlock::Create(M.getContext(), "", callRegFunc);
  Builder.SetInsertPoint(regEntry);

  llvm::SmallVector<llvm::Value *> ZeroZeroIndices{
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), llvm::APInt{32, 0}),
      llvm::ConstantInt::get(llvm::Type::getInt32Ty(M.getContext()), llvm::APInt{32, 0})};
  
  llvm::Value *strArg = Builder.CreateGEP(hcfContent->getValueType(), hcfContent, ZeroZeroIndices);
  llvm::Value *sizeArg = Builder.CreateLoad(i64Ty, hcfSize, "hcf_size");
  Builder.CreateCall(registerFunc, {strArg, sizeArg});
  Builder.CreateRetVoid();

  // Generate void @__acpp_invoke_unregister_hcf()
  llvm::FunctionType *callUnregFuncType = llvm::FunctionType::get(voidTy, false);
  llvm::Function *callUnregFunc = llvm::Function::Create(
      callUnregFuncType, llvm::GlobalValue::InternalLinkage, InvokeUnregisterHcfName, &M);

  llvm::BasicBlock *unregEntry = llvm::BasicBlock::Create(M.getContext(), "", callUnregFunc);
  Builder.SetInsertPoint(unregEntry);

  llvm::Value *idArg = Builder.CreateLoad(i64Ty, hcfId, "hcf_id");
  Builder.CreateCall(unregisterFunc, {idArg});
  Builder.CreateRetVoid();

  return true;
}

}

llvm::PreservedAnalyses HcfRegistrationPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &MAM) {
  if(generateCallerFunctions(M)) {

    auto* InvokeRegister = M.getFunction(InvokeRegisterHcfName);
    auto* InvokeUnregister = M.getFunction(InvokeUnregisterHcfName);

    if(InvokeRegister && InvokeUnregister) {
      makeFunctionPtrDiscoverablePerSection(M, InvokeRegister);

      modifyGlobalList(M, "llvm.global_ctors", InvokeRegister, 0, true);
      modifyGlobalList(M, "llvm.global_dtors", InvokeUnregister, 65535, false);
    }
  }

  M.print(llvm::outs(), nullptr);
  return llvm::PreservedAnalyses::none();
}

}