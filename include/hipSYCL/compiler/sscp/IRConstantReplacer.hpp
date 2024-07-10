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
#ifndef HIPSYCL_IR_CONSTANT_REPLACER_HPP
#define HIPSYCL_IR_CONSTANT_REPLACER_HPP

#include <cassert>
#include <climits>
#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/StringRef.h>
#include <type_traits>
#include <unordered_map>

#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Scalar/SCCP.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>

namespace hipsycl {
namespace compiler {

class IRConstant {
public:
  IRConstant(llvm::Module& Module, llvm::GlobalVariable& V)
  : M{&Module}, Var{&V} {}

  IRConstant()
  : M{nullptr}, Var{nullptr} {}

  bool isValid() const {
    return Var != nullptr && M != nullptr;
  }

  bool isInitialized() {
    if(!isValid())
      return false;
    
    if(!Var->hasInitializer())
      return false;
    
    if(Var->getName().find(".initialized") == std::string::npos)
      return false;
    
    return true;
  }

  // Set IR constant. Note that for std::string, this may replace the GlobalVariable
  // with a new one, so pointers the to global variable may be invalid
  // afterwards.
  template<class T>
  void set(T Value) {
    if(!isValid())
      return;
    
    assert(!isInitialized());
    auto Alignment = Var->getAlign();

    if constexpr(std::is_integral_v<T>){
      bool IsSigned = std::is_signed_v<T>;
      int Bits = sizeof(T) * CHAR_BIT;

      llvm::Constant *Initializer =
          llvm::ConstantInt::get(M->getContext(), llvm::APInt(Bits, Value, IsSigned));
      
      Var->setInitializer(Initializer);
    } else if constexpr(std::is_floating_point_v<T>) {
      llvm::Constant *Initializer = nullptr;
      if constexpr (std::is_same_v<float, T>) {
        Initializer = llvm::ConstantFP::get(llvm::Type::getFloatTy(M->getContext()), Value);
      } else {
        Initializer = llvm::ConstantFP::get(llvm::Type::getDoubleTy(M->getContext()), Value);
      }
      Var->setInitializer(Initializer);
    } else if constexpr(std::is_same_v<T, std::string>) {

      llvm::Constant *Initializer = llvm::ConstantDataArray::getRaw(
          Value + '\0', Value.size() + 1, llvm::Type::getInt8Ty(M->getContext()));

      // string case in general is more complicated because we expect
      // that strings can change size. This changes the type of the global
      // variable, since string length enters the LLVM type.
      if(Initializer->getType() != Var->getValueType()) {
        std::string Name = std::string{Var->getName()};

        // 1.) The idea is to create a new global variable of the appropriate type
        llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(*M, Initializer->getType(), true,
                                                                llvm::GlobalValue::InternalLinkage,
                                                                Initializer, Name + ".initialized");
        // 2.) Replace all uses of the old var with the new one
        //     We annot use replaceAllUsesWith because it requires same type, so
        //     we first have to create a cast
        llvm::Value* V = llvm::ConstantExpr::getPointerCast(NewVar, Var->getType());
        Var->replaceAllUsesWith(V);
        Var->eraseFromParent();
        
        Var = NewVar; 
      } else {
        Var->setInitializer(Initializer);
      }
      
    } else {
      M->getContext().emitError("Attempted setting hipSYCL IR constant of unsupported type");
    }
    
    if(Var->getName().find(".initialized") == std::string::npos) {
      Var->setName(Var->getName()+".initialized");
    }

    Var->setAlignment(Alignment);
    Var->setConstant(true);
    Var->setExternallyInitialized(false);
    Var->setLinkage(llvm::GlobalValue::InternalLinkage);
  }

  void set(const void* Buffer) {
    if(!isValid())
      return;

    if(Var->getValueType()->isIntegerTy(8)) {
      set(bit_cast<int8_t>(Buffer));
    } else if(Var->getValueType()->isIntegerTy(16)) {
      set(bit_cast<int16_t>(Buffer));
    } else if(Var->getValueType()->isIntegerTy(32)) {
      set(bit_cast<int32_t>(Buffer));
    } else if(Var->getValueType()->isIntegerTy(64)) {
      set(bit_cast<int64_t>(Buffer));
    } else if(Var->getValueType()->isFloatTy()) {
      set(bit_cast<float>(Buffer));
    } else if(Var->getValueType()->isDoubleTy()) {
      set(bit_cast<double>(Buffer));
    } else {
      M->getContext().emitError(
          "Attempted setting hipSYCL IR constant from buffer of unsupported type");
    }
  }

  llvm::GlobalVariable* getGlobalVariable() const {
    return Var;
  }

  // This should be executed once all IR constants have been set.
  static void optimizeCodeAfterConstantModification(llvm::Module &M,
                                                    llvm::ModuleAnalysisManager &MAM) {
    auto PromoteAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::PromotePass{});
    auto SCCPAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::SCCPPass{});
    auto ADCEAdaptor = llvm::createModuleToFunctionPassAdaptor(llvm::ADCEPass{});

    PromoteAdaptor.run(M, MAM);
    SCCPAdaptor.run(M, MAM);
    ADCEAdaptor.run(M, MAM);
    // This is necessary to remove backend-specific function definitions
    // that might no longer be needed after IR constant application.
    // In particular on the host side, where the kernel outlining pass
    // does not run.
    llvm::GlobalDCEPass GDCE;
    GDCE.run(M, MAM);
  }

protected:
  llvm::Module* M;
  llvm::GlobalVariable* Var;
private:
  template <class ToT>
  ToT bit_cast(const void* src) noexcept
  {
    ToT dst;
    memcpy(&dst, src, sizeof(ToT));
    return dst;
  }
};

class S2IRConstant : public IRConstant {
public:
  S2IRConstant() = default;
  S2IRConstant(llvm::Module& M, llvm::GlobalVariable& V)
  : IRConstant(M, V) {}

  template<class F>
  static void forEachS2IRConstant(llvm::Module& M, F&& Handler) {
    for(auto& V : M.globals()) {
      llvm::StringRef Name = V.getName();
      if (isS2IRConstantName(Name)) {
        Handler(S2IRConstant{M, V});
      }
    }
  }

  static S2IRConstant getFromConstantName(llvm::Module& M, const std::string& IrConstantName) {
    for(auto& V: M.globals()) {
      llvm::StringRef Name = V.getName();
      if(isS2IRConstantName(Name)) {
        if(Name.contains(IrConstantName))
          return S2IRConstant{M, V};
      }
    }
    return S2IRConstant{};
  }

  static S2IRConstant getFromFullName(llvm::Module& M, const std::string& FullName) {
    for(auto& V : M.globals()) {
      llvm::StringRef Name = V.getName();
      if(Name == FullName)
        return S2IRConstant{M, V};
    }
    return S2IRConstant{};
  }

  static void setCurrentBackend(llvm::Module& M);

private:
  static bool isS2IRConstantName(llvm::StringRef Name) {
    return Name.contains("__acpp_sscp_s2_ir_constant") &&
           Name.contains("__acpp_ir_constant_v");
  }
};


class S1IRConstantReplacer : public llvm::PassInfoMixin<S1IRConstantReplacer> {
public:
  S1IRConstantReplacer(const std::unordered_map<std::string, int> &IntConstants,
                       const std::unordered_map<std::string, uint64_t> &UInt64Constants,
                       const std::unordered_map<std::string, std::string> &StringConstants = {});

  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);

private:
  std::unordered_map<std::string, int> IntConstants;
  std::unordered_map<std::string, uint64_t> UInt64Constants;
  std::unordered_map<std::string, std::string> StringConstants;
};

}
}

#endif
