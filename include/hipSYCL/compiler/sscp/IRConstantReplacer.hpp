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

#ifndef HIPSYCL_IR_CONSTANT_REPLACER_HPP
#define HIPSYCL_IR_CONSTANT_REPLACER_HPP

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
    
    return Var->hasInitializer();
  }

  template<class T>
  void set(T Value) {
    if(!isValid())
      return;

    Var->setConstant(true);
    Var->setExternallyInitialized(false);
    Var->setLinkage(llvm::GlobalValue::InternalLinkage);

    llvm::Constant* Initializer = nullptr;
    if constexpr(std::is_integral_v<T>){
      bool IsSigned = std::is_signed_v<T>;
      int Bits = sizeof(T) * CHAR_BIT;
      Initializer = llvm::ConstantInt::get(M->getContext(), llvm::APInt(Bits, Value, IsSigned));
    } else if constexpr(std::is_floating_point_v<T>) {
      if constexpr (std::is_same_v<float, T>) {
        Initializer = llvm::ConstantFP::get(llvm::Type::getFloatTy(M->getContext()), Value);
      } else {
        Initializer = llvm::ConstantFP::get(llvm::Type::getDoubleTy(M->getContext()), Value);
      }
    } else if constexpr(std::is_same_v<T, std::string>) {
      llvm::StringRef RawData {Value};
      Initializer = llvm::ConstantDataArray::getRaw(RawData, Value.size(),
                                                    llvm::Type::getInt8Ty(M->getContext()));
    } else {
      M->getContext().emitError("Attempted setting hipSYCL IR constant of unsupported type");
    }
    
    if(Initializer) {
      Var->setInitializer(Initializer);
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
  }

protected:
  llvm::Module* M;
  llvm::GlobalVariable* Var;
};

class S2IRConstant : public IRConstant {
public:
  S2IRConstant() = default;
  S2IRConstant(llvm::Module& M, llvm::GlobalVariable& V)
  : IRConstant(M, V) {}

  template<class F>
  static void forEachS2IRConstant(llvm::Module& M, F&& Handler) {
    for(auto& V : M.getGlobalList()) {
      llvm::StringRef Name = V.getName();
      if (isS2IRConstantName(Name)) {
        Handler(S2IRConstant{M, V});
      }
    }
  }

  static S2IRConstant getFromConstantName(llvm::Module& M, const std::string& IrConstantName) {
    for(auto& V: M.getGlobalList()) {
      llvm::StringRef Name = V.getName();
      if(isS2IRConstantName(Name)) {
        if(Name.contains(IrConstantName))
          return S2IRConstant{M, V};
      }
    }
    return S2IRConstant{};
  }

  static S2IRConstant getFromFullName(llvm::Module& M, const std::string& FullName) {
    for(auto& V : M.getGlobalList()) {
      llvm::StringRef Name = V.getName();
      if(Name == FullName)
        return S2IRConstant{M, V};
    }
    return S2IRConstant{};
  }

  static void setCurrentBackend(llvm::Module& M);

private:
  static bool isS2IRConstantName(llvm::StringRef Name) {
    return Name.contains("__hipsycl_sscp_s2_ir_constant") &&
           Name.contains("__hipsycl_ir_constant_v");
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
