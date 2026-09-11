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
#include "hipSYCL/compiler/sscp/IRConstantReplacer.hpp"

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/Constant.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/GlobalVariable.h>


namespace hipsycl {
namespace compiler {

namespace {


void createIntConstant(const std::string& Name, int Bits, bool Signed, llvm::Module& M) {
  llvm::Constant *Initializer =
          llvm::ConstantInt::get(M.getContext(), llvm::APInt(Bits, 0, Signed));
  llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(
      M, Initializer->getType(), false, llvm::GlobalValue::ExternalLinkage, Initializer, Name);
}

void createInt32Constant(const std::string& Name, llvm::Module& M) {
  createIntConstant(Name, 32, true, M);
}

void createUInt64Constant(const std::string& Name, llvm::Module& M) {
  createIntConstant(Name, 64, false, M);
}

void createStrConstant(const std::string& Name, llvm::Module& M) {
  llvm::Constant *Initializer = llvm::ConstantDataArray::getRaw(
          "", 0, llvm::Type::getInt8Ty(M.getContext()));

  llvm::GlobalVariable *NewVar = new llvm::GlobalVariable(M, Initializer->getType(), false,
                                                                llvm::GlobalValue::ExternalLinkage,
                                                                Initializer, Name);
}

}

S1IRConstantReplacer::S1IRConstantReplacer(
    const std::unordered_map<std::string, int> &IntConstants,
    const std::unordered_map<std::string, uint64_t> &UInt64Constants,
    const std::unordered_map<std::string, std::string> &StringConstants, bool CreateIfUnused)
    : IntConstants{IntConstants}, UInt64Constants{UInt64Constants},
      StringConstants{StringConstants}, ForceCreation{CreateIfUnused} {}

llvm::PreservedAnalyses S1IRConstantReplacer::run(llvm::Module &M,
                                                  llvm::ModuleAnalysisManager &MAM) {
  auto setConstants = [&](auto CreateConstant, const auto& ConstantReplacementTable) {
    for(const auto& IC : ConstantReplacementTable) {
      llvm::GlobalVariable* G = M.getGlobalVariable(IC.first, true);
      
      if(!G && ForceCreation) {
        CreateConstant(IC.first, M);
        G = M.getGlobalVariable(IC.first, true);
      }
      
      if(G) {
        IRConstant C{M, *G};
        C.set(IC.second);
      }
    }
  };

  setConstants(createInt32Constant, IntConstants);
  setConstants(createUInt64Constant, UInt64Constants);
  setConstants(createStrConstant, StringConstants);


  // TODO Make this more specific
  return llvm::PreservedAnalyses::none();
}
}
}
