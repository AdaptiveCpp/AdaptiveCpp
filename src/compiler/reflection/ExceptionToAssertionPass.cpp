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




#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/GlobalValue.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <string>
#include "hipSYCL/compiler/reflection/ExceptionToAssertionPass.hpp"
//#include "hipSYCL/compiler/utils/ProcessFunctionAnnotationsPass.hpp"
#include <llvm/IR/PassManager.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/Support/Alignment.h>


#include <iostream>


namespace hipsycl {
namespace compiler {

llvm::PreservedAnalyses hipsycl::compiler::ExceptionToAssertionPass::run(llvm::Module &M, llvm::ModuleAnalysisManager &AM) {
  
  return llvm::PreservedAnalyses::none();
}


}
}
